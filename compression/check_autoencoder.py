"""One untrained interface smoke check on an existing calibration snapshot.

Never trains or launches Flower. This is not a reconstruction-quality experiment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from compression.autoencoder import BlockAutoencoder, encode, decode, weight_bytes, load_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--client', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    root = args.collection.resolve()
    trajectory = json.loads((root / f'client-{args.client}/validation/trajectory.json').read_text())
    snapshot = trajectory['snapshots'][-1]
    entry = snapshot['full_delta']
    path = (root / entry['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Snapshot path escapes collection')
    if path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Calibration snapshot file hash/size mismatch')
    with np.load(path, allow_pickle=False) as archive:
        tensors = {name: archive[name] for name in archive.files}
    shapes = {name: value.shape for name, value in tensors.items()}
    if {k: list(v) for k, v in shapes.items()} != trajectory['shapes']:
        raise ValueError('Calibration shape mismatch')
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = BlockAutoencoder().eval()
    start = perf_counter()
    packet = encode(tensors, model)
    encode_seconds = perf_counter() - start
    decoder_payload = weight_bytes(model, decoder_only=True)
    decoder = load_weights(decoder_payload, decoder_only=True)
    start = perf_counter()
    reconstructed = decode(packet, decoder, shapes)
    decode_seconds = perf_counter() - start
    assert list(reconstructed) == sorted(tensors)
    assert all(reconstructed[k].shape == tensors[k].shape and np.isfinite(reconstructed[k]).all() for k in tensors)
    zeros = [k for k in tensors if not np.any(tensors[k])]
    assert all(not np.any(reconstructed[k]) for k in zeros)
    summary = {
        'status': 'passed', 'scope': 'untrained codec interface smoke check; no accuracy or useful reconstruction claim',
        'collection': str(root), 'snapshot': snapshot['id'],
        'snapshot_file_sha256': entry['sha256'], 'tensor_count': len(tensors),
        'coordinate_count': sum(v.size for v in tensors.values()),
        'raw_float32_bytes': sum(v.nbytes for v in tensors.values()),
        'packet_bytes_including_header_layout_scales_latents': len(packet),
        'decoder_setup_bytes_including_header_checksum': len(decoder_payload),
        'decoder_sha256': hashlib.sha256(decoder_payload).hexdigest(),
        'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds,
        'timing_scope': 'single CPU interface check, untrained weights, not matched performance evaluation',
        'flower_traffic_measured': False,
        'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                          ['compression/autoencoder.py', 'compression/check_autoencoder.py']},
    }
    out = Path('outputs') / ('ae-codec-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (out / 'report.md').write_text(
        '# Standalone autoencoder codec interface check\n\n'
        f"Passed on {len(tensors)} tensors / {summary['coordinate_count']:,} coordinates from `{snapshot['id']}`. "
        'Reused one hash-verified calibration update; no additional training.\n\n'
        f'Packet: {len(packet):,} bytes including layout, decoder identity, scales and float32 latents. '
        f'Decoder setup artifact: {len(decoder_payload):,} bytes/client including its checksum. '
        'These are codec artifacts, not measured Flower or network traffic.\n\n'
        'Weights are untrained. Finite outputs and correct shapes establish interface correctness; '
        'they do not establish reconstruction quality, FL accuracy, or runtime superiority.\n\n'
        'Literature context: [Chandar et al. (2021)](https://arxiv.org/abs/2108.05670) '
        'uses client-trained autoencoders and server decoders for update compression. '
        'Our tensor-local normalized 256-value blocks and six-value latents are an adaptation. '
        'Calibration, fitting, decoder setup and inference must be included in the later feasibility assessment. '
        'The next batch fits one frozen configuration per client before comparing held-out reconstruction.\n',
        encoding='utf-8')
    print(f"Passed interface check: {len(packet):,} packet bytes; {len(decoder_payload):,} decoder setup bytes/client.")
    print('Untrained model; no reconstruction-quality or FL performance conclusion.')
    print(f'Saved: {out.resolve()}')


if __name__ == '__main__':
    main()
