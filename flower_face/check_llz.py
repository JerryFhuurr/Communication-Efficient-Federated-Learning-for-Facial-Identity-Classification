"""Run matched Flower QSGD/LLZ-p=0 experiments and require exact model replay."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from flower_face.reproducibility import source_hash
from flower_face.run import default_config, launch
from flower_face.study import check_pair, collect, snapshot, write_json
from federated_compression import compression_settings


def compare(baseline, compressed):
    """Require full learning equivalence before reporting communication savings."""
    if (baseline['config']['compression'], compressed['config']['compression']) != ('qsgd', 'qsgd-llz'):
        raise RuntimeError('Expected QSGD baseline and QSGD+LLZ comparison')
    compression_settings(compressed['config'])
    if compressed['config'].get('llz-p', 0) != 0:
        raise ValueError('Exact QSGD/LLZ replay comparison requires llz-p=0')
    check_pair(baseline, compressed)
    if baseline['metadata']['communication_measurement'] != compressed['metadata']['communication_measurement']:
        raise RuntimeError('Communication measurement boundaries differ')
    def learning_replay(run):
        result = []
        for row in run['replay']:
            if not isinstance(row.get('train'), dict):
                raise RuntimeError('Learning replay has invalid training metrics')
            result.append(dict(row, train={
                key: value for key, value in row['train'].items()
                if not key.startswith(('qsgd_', 'llz_'))}))
        return result

    if learning_replay(baseline) != learning_replay(compressed) or baseline['best'] != compressed['best']:
        raise RuntimeError('Lossless integration mismatch: model hashes, learning metrics, or selected checkpoint differ')
    methods = {}
    for result in (baseline, compressed):
        methods[result['config']['compression']] = dict(output=result['output'],
            communication=result['communication'])
    return dict(exact_replay=True, rounds=len(baseline['replay']),
        initial_model_sha256=baseline['metadata']['initial_model_sha256'],
        final_model_sha256=baseline['replay'][-1]['model_sha256'],
        best_checkpoint=baseline['best'], methods=methods,
        upload_reduction_percent=100*(1-compressed['train_upload_bytes']/baseline['train_upload_bytes']),
        total_reduction_percent=100*(1-compressed['total_bytes']/baseline['total_bytes']),
        measurement='Logical per-recipient serialized Flower objects, including codec/record metadata; not captured network traffic.',
        scope='Exact learning replay for this seed, configuration, source, environment, and number of rounds. Held-out test skipped.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--qsgd-levels', type=int, default=127)
    parser.add_argument('--llz-window', type=int, default=128)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or not 0 <= args.seed < 2**32:
        parser.error('rounds must be positive and seed must be in [0, 2**32-1]')
    root = Path(__file__).resolve().parents[1]
    base = default_config(root)
    base.update({'num-server-rounds': args.rounds, 'seed': args.seed,
                 'qsgd-levels': args.qsgd_levels, 'llz-p': 0, 'llz-window': args.llz_window,
                 'evaluate-final-test': False})
    try:
        compression_settings(dict(base, compression='qsgd-llz'))
    except ValueError as error:
        parser.error(str(error))
    output = ((args.output_dir or root/'outputs')/
              ('llz-flower-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = output/'app'
    snapshot(root, stage)
    source = source_hash(stage)
    manifest = hashlib.sha256(Path(base['manifest']).read_bytes()).hexdigest()
    record = dict(status='running', source_sha256=source, manifest_sha256=manifest, runs=[])
    write_json(output/'comparison.json', record)
    print(f'Comparison directory: {output}', flush=True)
    try:
        for method in ('qsgd', 'qsgd-llz'):
            run_root = output/method
            run_root.mkdir()
            config = dict(base, compression=method)
            config['output-dir'] = run_root.as_posix()
            print(f'Starting {method}: {args.rounds} rounds, seed={args.seed}', flush=True)
            if launch(config, root=root, app_dir=stage, log_path=run_root/'terminal.log') != 0:
                raise RuntimeError(f'Flower failed: inspect {run_root / "terminal.log"}')
            result = collect(run_root, config)
            if result['metadata']['source_sha256'] != source or result['metadata']['manifest_sha256'] != manifest:
                raise RuntimeError('Source or manifest changed during the comparison')
            record['runs'].append(result)
            write_json(output/'comparison.json', record)
            print(f'Completed {method}: {result["total_bytes"]:,} serialized bytes', flush=True)
        record['summary'] = compare(*record['runs'])
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(output/'comparison.json', record)
        raise
    write_json(output/'comparison.json', record)
    summary = record['summary']
    print(f'Exact replay passed for all {summary["rounds"]} rounds, including selected checkpoint.', flush=True)
    print(f'Training upload saved: {summary["upload_reduction_percent"]:.2f}%; '
          f'total serialized bytes saved: {summary["total_reduction_percent"]:.2f}%. Not network traffic.', flush=True)
    print(f'Saved comparison: {output / "comparison.json"}', flush=True)


if __name__ == '__main__':
    main()
