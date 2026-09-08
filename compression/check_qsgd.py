"""Reproducible synthetic QSGD codec check, with exact packet sizes and errors."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from compression.qsgd import decode, encode, packet_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 7, 31, 127])
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    if args.size < 1 or args.seed < 0 or any(s < 1 or s > 65535 for s in args.levels):
        parser.error('size must be positive, seed nonnegative, and levels in [1, 65535]')
    source = np.random.default_rng(args.seed).normal(size=args.size).astype(np.float32)
    rows = []
    print('Synthetic float32 vector; one realization per setting. Codec bytes only, not Flower traffic.')
    print(' levels | raw bytes | packet bytes | raw/packet | relative squared error')
    for levels in args.levels:
        # Separate streams from data generation; reproducible for each level setting.
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, levels, 1]))
        packet = encode(source, levels=levels, rng=rng)
        decoded = decode(packet)
        row = packet_stats(packet)
        row['raw_to_encoded_ratio'] = source.nbytes / len(packet)
        row['relative_squared_error'] = float(np.sum((decoded.astype(np.float64) - source)**2)
                                             / np.sum(source.astype(np.float64)**2))
        rows.append(row)
        print(f"{levels:7d} | {source.nbytes:9d} | {len(packet):12d} | "
              f"{row['raw_to_encoded_ratio']:10.3f} | {row['relative_squared_error']:.6f}")
    output = args.output_dir / ('qsgd-codec-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(mode='synthetic-codec-check', codec='qsgd-l2-dense-v1', seed=args.seed,
                  size=args.size, numpy_version=np.__version__, results=rows,
                  measurement='complete standalone packet, including metadata and padding; no Flower envelopes',
                  training_performed=False)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Saved codec check to {output.resolve()}')


if __name__ == '__main__':
    main()
