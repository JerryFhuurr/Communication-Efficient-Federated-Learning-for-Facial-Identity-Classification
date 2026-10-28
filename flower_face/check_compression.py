"""Matched three-method screening batch: QSGD, LLZ p=0 and LLZ p=1."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

from flower_face.check_llz import compare
from flower_face.compression_study import _check_lossy_pair, run_label
from flower_face.reproducibility import source_hash
from flower_face.run import default_config, launch
from flower_face.study import collect, snapshot, write_json

METHODS = ['qsgd', 'qsgd-llz-p0', 'qsgd-llz-p1']


def summarize(runs):
    labels = [run_label(r) for r in runs]
    if len(labels) != 3 or set(labels) != set(METHODS):
        raise ValueError('Exactly three distinct selected methods required')
    pair = dict(zip(labels, runs))
    exact = compare(pair['qsgd'], pair['qsgd-llz-p0'])
    _check_lossy_pair(pair['qsgd'], pair['qsgd-llz-p1'], 1)
    for run in runs:
        if run['metadata'].get('fixed_state_policy') != pair['qsgd']['metadata'].get('fixed_state_policy'):
            raise ValueError('Fixed-buffer policies differ')
    baseline = pair['qsgd']
    return dict(exact_p0_replay=exact, methods={name: dict(
        output=pair[name]['output'], selected_checkpoint=pair[name]['best'],
        final_validation=pair[name]['final_validation'], upload_bytes=pair[name]['train_upload_bytes'],
        total_serialized_bytes=pair[name]['total_bytes'],
        upload_reduction_vs_qsgd_percent=100*(1-pair[name]['train_upload_bytes']/baseline['train_upload_bytes']),
        total_reduction_vs_qsgd_percent=100*(1-pair[name]['total_bytes']/baseline['total_bytes']),
        selected_accuracy_difference_vs_qsgd_pp=100*(pair[name]['best']['validation_accuracy']-baseline['best']['validation_accuracy']))
        for name in METHODS}, scope='One matched seed, fixed model/data/FedAvg, no feedback, no held-out test; logical serialized bytes, not network traffic.')


def report(folder, record):
    summary = record['summary']
    lines = ['# Matched QSGD and LLZ comparison', '', summary['scope'], '',
             '| Method | Selected round | Validation accuracy | Loss | Upload MB | Total MB | Launch wall seconds |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in METHODS:
        row = summary['methods'][name]
        best = row['selected_checkpoint']
        lines.append(f"| {name} | {best['step']} | {best['validation_accuracy']:.2%} | {best['validation_loss']:.6f} | {row['upload_bytes']/1e6:.3f} | {row['total_serialized_bytes']/1e6:.3f} | {record['launch_seconds'][name]:.3f} |")
    lines += ['', f"Exact QSGD/LLZ p=0 replay passed for all {summary['exact_p0_replay']['rounds']} rounds and the selected checkpoint.",
        'Checkpoint selection minimizes validation loss, earliest tie wins. All-round byte totals are reported even when a selected checkpoint is earlier.', '',
        'Elapsed time surrounds each complete Flower launch: packaging/startup, simulation, training, codec work, validation and checkpoint writing. '
        'It excludes result collection/audit and report generation. Sequential fixed method order, one run/method; not an isolated codec benchmark or statistical speedup claim.', '',
        'Dataset, initialization, source snapshot, software, platform, fixed-buffer policy and settings are checked. The test split is unused. '
        'No uncompressed run, Top-k run, optimizer change, extra seed or tuning was included.', '',
        'Logical per-recipient serialized bytes include codec and Flower metadata. Downloads remain uncompressed. '
        'See comparison.json for configuration/provenance and summary.json for paired differences.', '',
        '[Post-run literature discussion](discussion.md).', '']
    (folder/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for run in record['runs']:
        name = run_label(run)
        style = '--' if name.endswith('p0') else '-'
        axes[0].plot([r['round'] for r in run['replay']], [100*r['validation']['eval_acc'] for r in run['replay']], label=name, linestyle=style)
        row = summary['methods'][name]
        axes[1].scatter(row['upload_bytes']/1e6, 100*row['selected_checkpoint']['validation_accuracy'], label=name,
                        marker='x' if name.endswith('p0') else 'o', s=70)
    axes[0].set(xlabel='Round', ylabel='Validation accuracy (%)', title='Matched seed 42; p=0 exactly overlaps QSGD')
    axes[1].set(xlabel='All-round training upload MB', ylabel='Selected validation accuracy (%)', title='Accuracy versus serialized upload cost')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.savefig(folder/'comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('outputs/finetune-compression-seed42/compression-study-20260925T091315807360Z/study.json'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.resume:
        folder = args.resume.resolve()
        record = json.loads((folder/'comparison.json').read_text())
        stage = folder/'app'
        if source_hash(stage) != record['source_sha256']:
            raise ValueError('Frozen source changed')
        if record['platform'] != platform.platform():
            raise ValueError('Platform changed since batch started')
        for saved in record['runs']:
            if collect(Path(saved['output']).parent, saved['config']) != saved:
                raise ValueError('Completed result changed')
        if record['status'] == 'completed':
            print(f'Completed valid batch reused: {folder}', flush=True)
            return
    else:
        previous = json.loads(args.reference.read_text())
        config = default_config(root)
        config.update(previous['protocol']['base_config'])
        config.update(compression='qsgd', **{'llz-p':0,'evaluate-final-test':False})
        if (config['seed'], config['num-server-rounds'], config['num-clients'], config['qsgd-levels'], config['llz-window']) != (42,15,4,127,128):
            raise ValueError('Expected fixed seed-42, 15-round, four-client, levels-127/window-128 screening batch')
        folder = root/'outputs'/('llz-matched-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        folder.mkdir()
        stage = folder/'app'
        snapshot(root, stage)
        record = dict(status='running', config=config, source_sha256=source_hash(stage),
            manifest_sha256=hashlib.sha256(Path(config['manifest']).read_bytes()).hexdigest(),
            platform=platform.platform(), reference=str(args.reference.resolve()), methods=METHODS,
            expected_initial_model_sha256=previous['runs'][0]['metadata']['initial_model_sha256'],
            runs=[], launch_seconds={}, attempts=[])
    if not folder.is_relative_to(root/'outputs'):
        raise ValueError('Results must stay under outputs/')
    if hashlib.sha256(Path(record['config']['manifest']).read_bytes()).hexdigest() != record['manifest_sha256']:
        raise ValueError('Manifest changed')
    print(f'Comparison directory: {folder}', flush=True)
    record['status'] = 'running'
    write_json(folder/'comparison.json', record)
    try:
        for name, p in [('qsgd',0), ('qsgd-llz-p0',0), ('qsgd-llz-p1',1)]:
            if any(run_label(r) == name for r in record['runs']):
                continue
            attempt_number = sum(a['method'] == name for a in record['attempts']) + 1
            run_root = folder/f'{name}-attempt-{attempt_number}'
            run_root.mkdir()
            config = dict(record['config'], compression='qsgd' if name=='qsgd' else 'qsgd-llz',
                          **{'llz-p':p,'output-dir':run_root.as_posix()})
            attempt = dict(method=name, output=str(run_root), status='running')
            record['attempts'].append(attempt)
            write_json(folder/'comparison.json', record)
            print(f'Starting {name}: 15 rounds, seed=42', flush=True)
            started = perf_counter()
            result_code = launch(config, root=root, app_dir=stage, log_path=run_root/'terminal.log')
            duration = perf_counter()-started
            if result_code:
                raise RuntimeError(f'Flower failed ({result_code}): {run_root / "terminal.log"}')
            result = collect(run_root, config)
            if (result['metadata']['source_sha256'] != record['source_sha256']
                    or result['metadata']['manifest_sha256'] != record['manifest_sha256']
                    or result['metadata']['initial_model_sha256'] != record['expected_initial_model_sha256']):
                raise ValueError('Source, manifest or initialization mismatch')
            if p == 0 and name != 'qsgd':
                compare(record['runs'][0], result)
                print('Exact p=0 replay passed for all 15 rounds and selected checkpoint.', flush=True)
            if p == 1:
                _check_lossy_pair(record['runs'][0], result, 1)
            record['runs'].append(result)
            record['launch_seconds'][name] = duration
            attempt['status'] = 'completed'
            write_json(folder/'comparison.json', record)
            print(f"Completed {name}: accuracy={result['best']['validation_accuracy']:.2%}, uploads={result['train_upload_bytes']:,} bytes, elapsed={duration:.1f}s", flush=True)
        record['summary'] = summarize(record['runs'])
        write_json(folder/'summary.json', record['summary'])
        report(folder, record)
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(folder/'comparison.json', record)
        raise
    write_json(folder/'comparison.json', record)
    print(f'Completed matched comparison: {folder}', flush=True)


if __name__ == '__main__':
    main()
