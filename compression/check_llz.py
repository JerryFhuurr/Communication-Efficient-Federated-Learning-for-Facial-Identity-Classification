"""Compare complete QSGD and QSGD+LLZ-p packets on identical quantized inputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from compression import llz_p, qsgd, qsgd_llz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--levels', type=int, nargs='+', default=[3, 127])
    parser.add_argument('--p', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--window-size', type=int, default=128)
    parser.add_argument('--input', type=Path, help='Optional float32/64 .npy array; no pickle loading')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    if not 1 <= args.size <= llz_p.MAX_SYMBOLS or args.seed < 0:
        parser.error('size must be in [1, 10000000] and seed nonnegative')
    if not 1 <= args.window_size <= 65535 or any(not 1 <= s <= 65535 for s in args.levels):
        parser.error('window-size and levels must be in [1, 65535]')
    if any(not 0 <= p <= 2*min(args.levels) for p in args.p):
        parser.error('p must be an integer in [0, 2*min(levels)]')
    if args.input:
        cases = {'input': np.load(args.input, allow_pickle=False)}
        if cases['input'].size > llz_p.MAX_SYMBOLS:
            parser.error('input exceeds default decoder symbol limit')
    else:
        rng = np.random.default_rng(args.seed)
        dense = rng.normal(size=args.size).astype(np.float32)
        cases = dict(gaussian=dense, sparse=dense*(rng.random(args.size)<0.1),
                     repeated=np.resize(np.array([0, 1, -1, 2, -2, 0, 0, 1], dtype=np.float32), args.size),
                     zeros=np.zeros(args.size, dtype=np.float32))
    output = args.output_dir/('llz-codec-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output/'inputs.npz', **cases)
    rows = []
    print('Standalone codec bytes including metadata/padding; no training or Flower traffic.')
    print('case       levels  p  QSGD bytes  QSGD+LLZ bytes  saved %  max code error')
    for index, (name, source) in enumerate(cases.items()):
        for levels in args.levels:
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, index, levels]))
            q = qsgd.quantize(source, levels=levels, rng=rng)  # Once per pair of methods.
            dense_packet = qsgd.pack(q)
            dense_reconstruction = qsgd.dequantize(q)
            (output/f'{name}-s{levels}-qsgd.bin').write_bytes(dense_packet)
            np.save(output/f'{name}-s{levels}-codes.npy', q.codes, allow_pickle=False)
            for p in args.p:
                start = perf_counter()
                packet = qsgd_llz.pack(q, p=p, window_size=args.window_size)
                encode_seconds = perf_counter()-start
                start = perf_counter()
                restored = qsgd_llz.unpack(packet)
                decoded = qsgd.dequantize(restored)
                decode_seconds = perf_counter()-start
                code_error = np.abs(restored.codes.astype(np.int64)-q.codes)
                maximum = int(code_error.max(initial=0))
                exact_codes = restored.codes.tobytes() == q.codes.tobytes()
                exact_floats = decoded.tobytes() == dense_reconstruction.tobytes()
                if maximum > p or (p == 0 and not (exact_codes and exact_floats)):
                    raise RuntimeError('LLZ distortion or lossless round-trip check failed')
                if (restored.norm, restored.levels, restored.dtype, restored.codes.shape) != (q.norm, q.levels, q.dtype, q.codes.shape):
                    raise RuntimeError('Reconstruction metadata mismatch')
                secondary_error = decoded.astype(np.float64)-dense_reconstruction.astype(np.float64)
                total_error = decoded.astype(np.float64)-source.astype(np.float64)
                saved = 100*(1-len(packet)/len(dense_packet))
                stats = qsgd_llz.packet_stats(packet)
                if len(packet)*8 != stats['metadata_bytes']*8+stats['triplet_bits']+stats['padding_bits']:
                    raise RuntimeError('Packet bit accounting mismatch')
                packet_file = f'{name}-s{levels}-p{p}-llz.bin'
                (output/packet_file).write_bytes(packet)
                rows.append(dict(case=name, levels=levels, p=p, qsgd=qsgd.packet_stats(dense_packet),
                    qsgd_llz=stats, saved_percent=saved, max_code_error=maximum,
                    hamming_error=float(np.count_nonzero(code_error)/q.codes.size) if q.codes.size else 0.0,
                    exact_codes=exact_codes, exact_qsgd_reconstruction=exact_floats,
                    secondary_squared_error=float(np.sum(secondary_error**2)),
                    total_squared_error=float(np.sum(total_error**2)),
                    encode_seconds=encode_seconds, decode_seconds=decode_seconds,
                    qsgd_packet_sha256=hashlib.sha256(dense_packet).hexdigest(),
                    llz_packet_sha256=hashlib.sha256(packet).hexdigest(), packet=packet_file))
                print(f'{name:10} {levels:6d} {p:2d} {len(dense_packet):11d} {len(packet):15d} {saved:8.2f} {maximum:15d}')
    package = Path(__file__).resolve().parent
    report = dict(mode='input-codec-check' if args.input else 'synthetic-codec-check',
        seed=args.seed, window_size=args.window_size, numpy_version=np.__version__, results=rows,
        input_path=str(args.input.resolve()) if args.input else None,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (package/'llz_p.py', package/'qsgd_llz.py', package/'qsgd.py', Path(__file__))},
        measurement='Complete standalone packets including headers, quantization metadata and padding. No Flower envelopes or network traffic.',
        note='One quantization per input/level reused by both codecs. Synthetic data is not a measured client update. Single timings are diagnostic, not a performance study.',
        training_performed=False)
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Saved codec comparison to {output.resolve()}')


if __name__ == '__main__':
    main()
