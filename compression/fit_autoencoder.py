"""Fixed four-client AE fitting and held-out codec probes; no Flower training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
from time import perf_counter

import numpy as np
import torch

from compression import autoencoder as ae, qsgd, topk

CONFIG = dict(epochs=10, batch_size=1024, learning_rate=0.001,
              seed=4242, qsgd_levels=127, topk_k=124000,
              architecture=[256, 64, 6, 64, 256], feedback=False)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_path(root, entry):
    path = (root / entry['path']).resolve()
    if not path.is_relative_to(root) or path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']:
        raise ValueError('Calibration artifact path, size or hash mismatch')
    return path


def read_samples(root, trajectory):
    with np.load(verified_path(root, trajectory['sampled_file']), allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    x, scales, weights, valid = (data[k] for k in ['blocks', 'scales', 'population_weights', 'valid_length'])
    n = len(x)
    if (x.shape != (n, 256) or x.dtype != np.float32 or n < 1
            or any(a.shape != (n,) for a in [scales, weights, valid])
            or not all(np.isfinite(a).all() for a in [x, scales, weights])
            or np.any(scales < 0) or np.any(weights <= 0)
            or valid.dtype.kind not in 'iu' or np.any(valid < 1) or np.any(valid > 256)):
        raise ValueError('Invalid calibration samples')
    mask = np.arange(256)[None, :] < valid[:, None]
    if np.any(x[~mask] != 0) or np.any(x[scales == 0] != 0):
        raise ValueError('Invalid padding or zero normalization')
    return dict(x=torch.from_numpy(x), scales=torch.from_numpy(scales),
                weights=torch.from_numpy(weights.astype(np.float64)),
                mask=torch.from_numpy(mask))


def squared_errors(prediction, data, indices):
    """Masked block SSE; zero-norm blocks follow the actual codec bypass."""
    prediction = torch.where(data['scales'][indices, None] == 0, 0., prediction)
    return ((prediction - data['x'][indices]).square() * data['mask'][indices]).sum(dim=1)


def validation_loss(model, data):
    numerator = 0.
    denominator = float((data['weights'] * data['mask'].sum(1)).sum())
    with torch.inference_mode():
        for start in range(0, len(data['x']), 1024):
            indices = slice(start, start + 1024)
            numerator += float((squared_errors(model(data['x'][indices]), data, indices).double()
                                * data['weights'][indices]).sum())
    return numerator / denominator


def fit_client(client, fit, validation, out, signature):
    directory = out / f'client-{client}'
    directory.mkdir(exist_ok=True)
    record_path = directory / 'fit.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if (record['signature'] != signature or file_hash(directory / 'model.aew') != record['model_sha256']
                or file_hash(directory / 'decoder.aew') != record['decoder_sha256']):
            raise ValueError('Existing fitting artifact is incompatible or corrupted')
        return ae.load_weights((directory / 'model.aew').read_bytes()), record
    seed = CONFIG['seed'] + client
    torch.manual_seed(seed)
    model = ae.BlockAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    generator = torch.Generator().manual_seed(seed)
    # Fixed global denominator makes shuffled minibatch gradients an unbiased
    # estimate of the population-weighted masked objective, including tails.
    normalizer = float((fit['weights'] * fit['mask'].sum(1)).mean())
    history = []
    best = math.inf
    started = perf_counter()
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        order = torch.randperm(len(fit['x']), generator=generator)
        total = 0.
        for indices in order.split(CONFIG['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            errors = squared_errors(model(fit['x'][indices]), fit, indices)
            loss = (errors.double() * fit['weights'][indices]).mean() / normalizer
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fitting loss')
            loss.backward()
            optimizer.step()
            total += float((errors.detach().double() * fit['weights'][indices]).sum())
        model.eval()
        value = validation_loss(model, validation)
        history.append(dict(epoch=epoch, training_online_mse=total / (normalizer * len(order)),
                            validation_normalized_masked_mse=value))
        if value < best:
            best = value
            selected = epoch
            (directory / 'model.aew').write_bytes(ae.weight_bytes(model))
    model = ae.load_weights((directory / 'model.aew').read_bytes())
    (directory / 'decoder.aew').write_bytes(ae.weight_bytes(model, decoder_only=True))
    record = dict(signature=signature, client=client, seed=seed, selected_epoch=selected,
                  selected_validation_mse=best, history=history, fitting_seconds=perf_counter() - started,
                  fit_blocks=len(fit['x']), validation_blocks=len(validation['x']),
                  model_sha256=file_hash(directory / 'model.aew'),
                  decoder_sha256=file_hash(directory / 'decoder.aew'),
                  decoder_setup_bytes=(directory / 'decoder.aew').stat().st_size)
    write_json(record_path, record)
    return model, record


def probe(tensors, model, method, seed):
    names = sorted(tensors)
    shapes = {k: tensors[k].shape for k in names}
    _, layout = ae._layout(shapes)
    start = perf_counter()
    if method == 'autoencoder':
        packet = ae.encode(tensors, model)
    elif method == 'qsgd':
        rng = np.random.default_rng(seed)
        parts = [qsgd.encode(tensors[k], levels=CONFIG['qsgd_levels'], rng=rng) for k in names]
        packet = b'QBP1' + struct.pack('<I', len(layout)) + layout + b''.join(
            struct.pack('<I', len(part)) + part for part in parts)
    else:
        vector = np.concatenate([tensors[k].reshape(-1) for k in names])
        packet = b'TBP1' + struct.pack('<I', len(layout)) + layout + topk.encode(vector, CONFIG['topk_k'])
    encode_seconds = perf_counter() - start
    start = perf_counter()
    if method == 'autoencoder':
        restored = ae.decode(packet, model.decoder, shapes)
    else:
        if packet[8:8 + len(layout)] != layout:
            raise ValueError('Probe layout mismatch')
        offset = 8 + len(layout)
        restored = {}
        if method == 'qsgd':
            for name in names:
                length, = struct.unpack_from('<I', packet, offset)
                offset += 4
                restored[name] = qsgd.decode(packet[offset:offset + length])
                offset += length
            if offset != len(packet):
                raise ValueError('Probe bundle trailing bytes')
        else:
            vector = topk.decode(packet[offset:], expected_size=sum(tensors[k].size for k in names),
                                 expected_k=CONFIG['topk_k'])
            offset = 0
            for name in names:
                n = tensors[name].size
                restored[name] = vector[offset:offset + n].reshape(shapes[name])
                offset += n
    decode_seconds = perf_counter() - start
    error = energy = reconstructed_energy = dot = 0.
    per_tensor = {}
    for name in names:
        if restored[name].shape != shapes[name] or not np.isfinite(restored[name]).all():
            raise ValueError('Invalid reconstructed tensor')
        x, y = tensors[name].astype(np.float64), restored[name].astype(np.float64)
        e, norm = float(np.sum((x - y)**2)), float(np.sum(x*x))
        error += e; energy += norm
        reconstructed_energy += float(np.sum(y*y)); dot += float(np.sum(x*y))
        per_tensor[name] = dict(squared_error=e, energy=norm)
    return dict(method=method, packet_bytes=len(packet), packet_sha256=hashlib.sha256(packet).hexdigest(),
                squared_error=error, energy=energy, relative_squared_error=error / energy if energy else 0.,
                cosine=dot / math.sqrt(energy * reconstructed_energy) if energy and reconstructed_energy else None,
                encode_seconds=encode_seconds, decode_seconds=decode_seconds, per_tensor=per_tensor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Existing compatible directory permits completed-client reuse')
    args = parser.parse_args()
    root = args.collection.resolve()
    collection = json.loads((root / 'collection.json').read_text())
    if collection['status'] != 'completed' or collection['audit']['status'] != 'passed':
        raise ValueError('Completed audited calibration required')
    source = {name: file_hash(Path(name)) for name in ['compression/autoencoder.py', 'compression/block_data.py',
              'compression/fit_autoencoder.py', 'compression/qsgd.py', 'compression/topk.py']}
    protocol = dict(config=CONFIG, collection_sha256=file_hash(root / 'collection.json'), source_sha256=source,
                    environment=dict(python=platform.python_version(), platform=platform.platform(),
                                     torch=torch.__version__, numpy=np.__version__))
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    out = args.output or Path('outputs') / ('ae-feasibility-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out = out.resolve()
    if not out.is_relative_to((Path.cwd() / 'outputs').resolve()):
        raise ValueError('Generated results must stay under outputs/')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Output protocol mismatch')
        if (out / 'summary.json').exists():
            print(f'Completed batch already exists: {out}', flush=True)
            return
    write_json(out / 'protocol.json', protocol)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(f'Batch directory: {out}', flush=True)
    started = perf_counter()
    fits, results = [], []
    for client in range(4):
        trajectories = [json.loads((root / f'client-{client}/{role}/trajectory.json').read_text())
                        for role in ['fit', 'validation']]
        if trajectories != collection['clients'][client]['trajectories']:
            raise ValueError('Trajectory metadata differs from audited collection')
        if [t['role'] for t in trajectories] != ['fit', 'validation'] or trajectories[0]['seed'] == trajectories[1]['seed']:
            raise ValueError('Independent fitting/validation trajectories required')
        fit, validation = [read_samples(root, t) for t in trajectories]
        model, record = fit_client(client, fit, validation, out, signature)
        fits.append(record)
        print(f"Client {client}: fitted 10 epochs, selected {record['selected_epoch']}, validation MSE={record['selected_validation_mse']:.8f}", flush=True)
        del fit, validation
        for snapshot in trajectories[1]['snapshots']:
            path = verified_path(root, snapshot['full_delta'])
            with np.load(path, allow_pickle=False) as archive:
                tensors = {k: archive[k] for k in archive.files}
            if {k: list(v.shape) for k, v in tensors.items()} != trajectories[1]['shapes']:
                raise ValueError('Full delta shape mismatch')
            probe_path = out / f'client-{client}' / f"probe-{snapshot['step']:03d}.json"
            if probe_path.exists():
                rows = json.loads(probe_path.read_text())
                if any(r['signature'] != signature or r['snapshot_sha256'] != snapshot['full_delta']['sha256'] for r in rows):
                    raise ValueError('Existing probe mismatch')
            else:
                seed = CONFIG['seed'] + client * 1000 + snapshot['step']
                rows = [dict(probe(tensors, model, method, seed), client=client, snapshot=snapshot['id'],
                             snapshot_sha256=snapshot['full_delta']['sha256'], signature=signature, qsgd_seed=seed)
                        for method in ['autoencoder', 'qsgd', 'topk']]
                write_json(probe_path, rows)
            results.extend(rows)
        print(f'Client {client}: all 10 held-out updates probed with 3 codecs', flush=True)
    if len(results) != 120 or len({(r['snapshot'], r['method']) for r in results}) != 120:
        raise ValueError('Incomplete or duplicate probe batch')
    for name, digest in source.items():
        if file_hash(Path(name)) != digest:
            raise ValueError('Source changed during batch')
    summary = dict(status='completed', signature=signature, fits=fits, methods={},
                   elapsed_seconds=perf_counter() - started, calibration_seconds=collection['elapsed_seconds'],
                   audit=dict(status='passed', clients=4, held_out_updates=40, codec_probes=120,
                              identity_validation_or_test_used=False, finite_shapes_and_hashes_checked=True),
                   scope='Offline trajectory-held-out reconstruction; shared training images; no FL accuracy or network traffic')
    for method in ['autoencoder', 'qsgd', 'topk']:
        rows = [r for r in results if r['method'] == method]
        summary['methods'][method] = dict(updates=len(rows), packet_bytes_per_update=rows[0]['packet_bytes'],
            total_probe_bytes=sum(r['packet_bytes'] for r in rows),
            energy_weighted_relative_squared_error=sum(r['squared_error'] for r in rows) / sum(r['energy'] for r in rows),
            median_relative_squared_error=float(np.median([r['relative_squared_error'] for r in rows])),
            mean_cosine=float(np.mean([r['cosine'] for r in rows if r['cosine'] is not None])),
            median_encode_seconds=float(np.median([r['encode_seconds'] for r in rows])),
            median_decode_seconds=float(np.median([r['decode_seconds'] for r in rows])))
    setup = sum(r['decoder_setup_bytes'] for r in fits)
    summary['decoder_setup_bytes_four_clients'] = setup
    summary['codec_only_15_round_projection'] = {m: 60 * v['packet_bytes_per_update'] + (setup if m == 'autoencoder' else 0)
                                               for m, v in summary['methods'].items()}
    summary['codec_only_break_even_rounds'] = {}
    for reference in ['qsgd', 'topk']:
        saving = 4 * (summary['methods'][reference]['packet_bytes_per_update'] - summary['methods']['autoencoder']['packet_bytes_per_update'])
        summary['codec_only_break_even_rounds'][reference] = math.floor(setup / saving) + 1 if saving > 0 else None
    write_json(out / 'probes.json', results)
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary['methods'], indent=2), flush=True)
    print(f'Completed and audited: {out}', flush=True)


if __name__ == '__main__':
    main()
