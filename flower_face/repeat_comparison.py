"""Run exactly seeds 43/44 for five frozen methods, reusing seed 42."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

from flower_face.complete_comparison import METHODS, summarize
from flower_face.compression_study import file_hash, run_label
from flower_face.reproducibility import source_hash
from flower_face.run import launch
from flower_face.study import collect, snapshot, write_json

SEEDS = (43, 44)


def plan(base, output):
    """Only seeds and per-method codec switches differ from the frozen config."""
    if (base['seed'], base['num-server-rounds'], base['num-clients'],
            base['qsgd-levels'], base['llz-window'], base['topk-k'],
            base['evaluate-final-test']) != (42, 15, 4, 127, 128, 124000, False):
        raise ValueError('Expected frozen seed-42 five-method protocol')
    return [dict(base, seed=seed,
                 compression='qsgd-llz' if name.startswith('qsgd-llz-') else name,
                 **{'llz-p': int(name[-1]) if name.startswith('qsgd-llz-') else 0,
                    'output-dir': (output/f'seed-{seed}-{name}').as_posix()})
            for seed in SEEDS for name in METHODS]


def check_environment(metadata):
    current = dict(platform=platform.platform(), python=platform.python_version(),
                   versions={key: version(key) for key in metadata['versions']})
    if any(current[key] != metadata[key] for key in current):
        raise ValueError('Environment differs from frozen seed-42 reference')


def check_run(reference, candidate, expected):
    if candidate['config'] != expected:
        raise ValueError('Candidate configuration differs from the fixed plan')
    for key in ('manifest_sha256', 'source_sha256', 'versions', 'platform',
                'python', 'reproducibility', 'selection_rule',
                'communication_measurement', 'fixed_state_policy'):
        if candidate['metadata'][key] != reference['metadata'][key]:
            raise ValueError(f'Frozen protocol mismatch: {key}')
    # Initialization can depend on training seed. Each seed's five-method
    # summary checks that its initialization is identical across methods.


def audit_seed(seed, runs):
    return summarize([r for r in runs if r['config']['seed'] == seed])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('outputs/compression-controls-20261012T073109367046Z'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.resume:
        output = args.resume.resolve()
        record = json.loads((output/'batch.json').read_text())
    else:
        baseline = args.baseline.resolve()
        prior = json.loads((baseline/'comparison.json').read_text())
        if prior['status'] != 'completed' or len(prior['runs']) != 5:
            raise ValueError('Completed five-method seed-42 reference required')
        check_environment(prior['runs'][0]['metadata'])
        if source_hash(baseline/'app') != prior['source_sha256']:
            raise ValueError('Baseline app changed')
        output = root/'outputs'/('compression-repeats-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        configs = plan(prior['config'], output)
        output.mkdir()
        snapshot(baseline/'app', output/'app')
        record = dict(status='ready', baseline=str(baseline), source_sha256=prior['source_sha256'],
                      manifest_sha256=prior['manifest_sha256'], base_config=prior['config'],
                      plan=configs, reused_runs=prior['runs'], runs=[], attempts=[], seed_checks={})
        write_json(output/'batch.json', record)
        protocol = ['# Frozen repeated-seed batch, 16 October 2026', '',
            'Exactly ten runs: seeds 43 and 44, each with uncompressed FedAvg, QSGD, LLZ p=0, LLZ p=1 and Top-k+EF.',
            'Seed 42 is reused from the completed 10/12 October comparison. No tuning, test evaluation, new dataset, optimizer or extra seed.', '',
            '## Fixed settings', '', '```json', json.dumps(record['base_config'], indent=2), '```', '',
            'Only seed, compression/LLZ tolerance and output path change. Dataset split stays fixed.',
            'QSGD levels=127, LLZ window=128, Top-k k=124000. All four clients participate for 15 rounds.',
            'Source snapshot, manifest, environment, fixed-buffer policy, measurements and checkpoint selection remain fixed.',
            f"Frozen app SHA256: {record['source_sha256']}",
            f"Manifest SHA256: {record['manifest_sha256']}", '',
            'Within each seed, initialization is identical across all methods. Top-k alone has error feedback.',
            'Exact p=0 replay is checked as soon as that pair finishes, including all round hashes, learning metrics and selected checkpoint.',
            'Each completed run is collected once to reconcile participation, ledgers and checkpoints. Resume revalidates saved runs and skips completed methods.',
            'Runtime includes the complete Flower launch and remains descriptive. Bytes include metadata, with uncompressed downloads and no physical network capture.',
            'This batch saves per-run evidence and per-seed gates only. Across-seed statistical analysis and slide updates are separate later tasks.', '']
        (output/'protocol.md').write_text('\n'.join(protocol), encoding='utf-8')
    if not output.is_relative_to(root/'outputs'):
        raise ValueError('Batch must stay under outputs/')
    stage = output/'app'
    reference = record['reused_runs'][0]
    check_environment(reference['metadata'])
    if (source_hash(stage) != record['source_sha256']
            or file_hash(record['base_config']['manifest']) != record['manifest_sha256']):
        raise ValueError('Frozen source or manifest changed')
    if record['plan'] != plan(record['base_config'], output):
        raise ValueError('Saved plan differs from the fixed ten-run batch')
    # Existing valid completed runs are reused, never relaunched.
    for saved in record['reused_runs'] + record['runs']:
        current = collect(Path(saved['output']).parent, saved['config'])
        if current != saved:
            raise ValueError('Saved completed result changed')
    if record['status'] == 'completed':
        print(f'Reusing completed batch: {output}', flush=True)
        return
    record.update(status='running')
    record.pop('error', None)
    write_json(output/'batch.json', record)
    print(f'Frozen batch directory: {output}', flush=True)
    try:
        for config in record['plan']:
            seed, label = config['seed'], run_label(dict(config=config))
            if any(r['config']['seed']==seed and run_label(r)==label for r in record['runs']):
                continue
            run_root = Path(config['output-dir'])
            # Preserve a completed process result if interruption occurred before collection.
            completed = list(run_root.glob('*/metrics.json')) if run_root.exists() else []
            if completed:
                result = collect(run_root, config)
            else:
                if run_root.exists():
                    raise ValueError(f'Incomplete attempt retained at {run_root}; inspect before retrying')
                run_root.mkdir()
                if (source_hash(stage) != record['source_sha256']
                        or file_hash(config['manifest']) != record['manifest_sha256']):
                    raise ValueError('Frozen input changed before launch')
                check_environment(reference['metadata'])
                attempt = dict(seed=seed, method=label, status='running', output=str(run_root))
                record['attempts'].append(attempt)
                write_json(output/'batch.json', record)
                print(f'Starting seed={seed} {label}: 15 rounds', flush=True)
                started = perf_counter()
                code = launch(config, root=root, app_dir=stage, log_path=run_root/'terminal.log')
                attempt['launch_seconds'] = perf_counter()-started
                if code:
                    attempt.update(status='failed', exit_code=code)
                    raise RuntimeError(f'Flower failed: {run_root / "terminal.log"}')
                result = collect(run_root, config)
                attempt['status'] = 'completed'
            check_run(reference, result, config)
            record['runs'].append(result)
            if label == 'qsgd-llz-p0':
                from flower_face.check_llz import compare
                qsgd = next(r for r in record['runs'] if r['config']['seed']==seed and run_label(r)=='qsgd')
                exact = compare(qsgd, result)
                record['seed_checks'][str(seed)] = dict(exact_p0_replay=exact)
                print(f'Exact p=0 replay passed for seed={seed}, all 15 rounds and checkpoint', flush=True)
            write_json(output/'batch.json', record)
            print(f"Completed {len(record['runs'])}/10: seed={seed} {label}, accuracy={result['best']['validation_accuracy']:.2%}, uploads={result['train_upload_bytes']:,}", flush=True)
            if sum(r['config']['seed']==seed for r in record['runs']) == 5:
                checked = audit_seed(seed, record['runs'])
                record['seed_checks'][str(seed)]['five_method_compatibility'] = 'passed'
                write_json(output/f'seed-{seed}-audit.json', checked)
                write_json(output/'batch.json', record)
        if {(r['config']['seed'], run_label(r)) for r in record['runs']} != {(seed,name) for seed in SEEDS for name in METHODS} or len(record['runs']) != 10:
            raise ValueError('Incomplete or duplicate ten-run matrix')
        record.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(output/'batch.json', record)
        raise
    write_json(output/'batch.json', record)
    print(f'Completed all ten requested runs: {output}', flush=True)


if __name__ == '__main__':
    main()
