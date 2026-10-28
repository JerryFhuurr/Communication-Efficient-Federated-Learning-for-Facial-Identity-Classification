"""Add exactly two controls to a completed QSGD/LLZ batch, reusing its app."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
from time import perf_counter

from flower_face.check_compression import summarize as summarize_llz
from flower_face.check_topk import compatibility
from flower_face.compression_study import run_label, file_hash
from flower_face.reproducibility import source_hash
from flower_face.run import launch
from flower_face.study import check_pair, collect, snapshot, write_json

METHODS = ['none', 'qsgd', 'qsgd-llz-p0', 'qsgd-llz-p1', 'topk-ef']


def check_control(reference, candidate):
    compatibility(reference, candidate)
    for key in ['source_sha256', 'communication_measurement', 'python']:
        if reference['metadata'].get(key) != candidate['metadata'].get(key):
            raise ValueError(f'Control compatibility failed: {key}')
    check_pair(reference, candidate)
    if candidate['config']['compression'] not in ['none','topk-ef'] or candidate['config']['topk-k'] != 124000:
        raise ValueError('Expected uncompressed or k=124000 Top-k control')


def summarize(runs):
    labels = [run_label(r) for r in runs]
    if len(labels) != 5 or set(labels) != set(METHODS):
        raise ValueError('Exactly five distinct methods required')
    pair = dict(zip(labels, runs))
    llz = summarize_llz([pair[m] for m in METHODS[1:4]])
    for name in ['none', 'topk-ef']:
        check_control(pair['qsgd'], pair[name])
    base = pair['none']
    rows = {}
    for name in METHODS:
        run = pair[name]
        rows[name] = dict(output=run['output'], selected_checkpoint=run['best'],
            final_validation=run['final_validation'], upload_bytes=run['train_upload_bytes'],
            total_bytes=run['total_bytes'], feedback=name == 'topk-ef',
            accuracy_difference_vs_none_pp=100*(run['best']['validation_accuracy']-base['best']['validation_accuracy']),
            upload_saving_vs_none_percent=100*(1-run['train_upload_bytes']/base['train_upload_bytes']),
            total_saving_vs_none_percent=100*(1-run['total_bytes']/base['total_bytes']))
    return dict(methods=rows, exact_p0_replay=llz['exact_p0_replay'],
        topk_vs_p0=dict(upload_budget_difference_percent=100*(pair['topk-ef']['train_upload_bytes']/pair['qsgd-llz-p0']['train_upload_bytes']-1),
                       accuracy_difference_pp=100*(pair['topk-ef']['best']['validation_accuracy']-pair['qsgd-llz-p0']['best']['validation_accuracy'])),
        scope='One seed, identical frozen app/model/data/settings/environment. Top-k uses feedback; other methods do not. Method comparison, not an isolated codec ablation. Test unused; logical bytes, not physical network traffic.')


def report(folder, record):
    result = record['summary']
    lines = ['# Five-method compression screening comparison', '', result['scope'], '',
        '| Method | Feedback | Selected round | Accuracy | Loss | Upload MB | Total MB | Launch seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in METHODS:
        row = result['methods'][name]; best = row['selected_checkpoint']
        lines.append(f"| {name} | {'yes' if row['feedback'] else 'no'} | {best['step']} | {best['validation_accuracy']:.2%} | {best['validation_loss']:.6f} | {row['upload_bytes']/1e6:.3f} | {row['total_bytes']/1e6:.3f} | {record['launch_seconds'][name]:.3f} |")
    lines += ['', 'Lowest validation loss selects each checkpoint; all-round bytes cover the full 15 rounds. Exact p=0 replay includes all rounds and selected checkpoint.', '',
        'QSGD/LLZ results and elapsed times are reused from the audited 10 October batch. Only uncompressed and Top-k+EF controls were newly trained. '
        'All use the same frozen app source, software, Windows build, initialization and feature cache. Runs were made on different days; runtime is descriptive, '
        'with fixed method order, startup/simulation/training/codec/validation/checkpoint costs included. No isolated codec or statistical speedup claim.', '',
        'Top-k has persistent error feedback; QSGD/LLZ have none. A performance difference cannot be attributed solely to sparsification versus quantization. '
        'k=124000 was fixed before this control run to approximate the historical p=0 upload budget; no budget or accuracy tuning was done.', '',
        '[Literature interpretation](discussion.md) · [Meeting progress](meeting-progress.md)', '']
    (folder/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for run in record['runs']:
        name = run_label(run); row = result['methods'][name]
        axes[0].plot([r['round'] for r in run['replay']], [100*r['validation']['eval_acc'] for r in run['replay']],
                     label=name, linestyle='--' if name.endswith('p0') else '-')
        axes[1].scatter(row['upload_bytes']/1e6, 100*row['selected_checkpoint']['validation_accuracy'],
                        label=name, marker='x' if name.endswith('p0') else 'o', s=70)
    axes[0].set(xlabel='Round', ylabel='Validation accuracy (%)', title='Fixed seed 42; p=0 exactly overlaps QSGD')
    axes[1].set(xlabel='All-round training uploads (MB, log scale)', ylabel='Selected accuracy (%)', title='Communication versus accuracy; feedback differs')
    axes[1].set_xscale('log')
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(folder/'comparison.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('outputs/llz-matched-20261010T114122936542Z'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    if args.resume:
        folder = args.resume.resolve(); record = json.loads((folder/'comparison.json').read_text())
    else:
        baseline = args.baseline.resolve(); prior = json.loads((baseline/'comparison.json').read_text())
        if prior['status'] != 'completed': raise ValueError('Completed QSGD/LLZ batch required')
        folder = root/'outputs'/('compression-controls-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        if source_hash(baseline/'app') != prior['source_sha256']: raise ValueError('Frozen baseline source changed')
        reference = prior['runs'][0]
        if (reference['metadata']['platform'] != platform.platform()
                or reference['metadata']['python'] != platform.python_version()
                or reference['metadata']['versions'] != {k:version(k) for k in reference['metadata']['versions']}):
            raise ValueError('Current environment differs; no control training launched')
        folder.mkdir(); snapshot(baseline/'app', folder/'app')
        record = dict(status='running', baseline=str(baseline), source_sha256=prior['source_sha256'],
            config=prior['config'], manifest_sha256=prior['manifest_sha256'], runs=prior['runs'],
            launch_seconds=prior['launch_seconds'], attempts=[], reused_methods=METHODS[1:4])
    if not folder.is_relative_to(root/'outputs'): raise ValueError('Outputs must stay under outputs/')
    stage = folder/'app'; reference = next(r for r in record['runs'] if run_label(r)=='qsgd')
    if source_hash(stage) != record['source_sha256'] or file_hash(record['config']['manifest']) != record['manifest_sha256']:
        raise ValueError('Frozen source or manifest changed')
    if (reference['metadata']['platform'] != platform.platform()
            or reference['metadata']['python'] != platform.python_version()
            or reference['metadata']['versions'] != {k:version(k) for k in reference['metadata']['versions']}):
        raise ValueError('Environment differs from completed results')
    for saved in record['runs']:
        if collect(Path(saved['output']).parent, saved['config']) != saved:
            raise ValueError('Saved completed result changed')
    print(f'Control comparison directory: {folder}', flush=True)
    if record['status']=='completed': print('Reusing completed valid comparison.', flush=True); return
    write_json(folder/'comparison.json', record)
    try:
        for name in ['none','topk-ef']:
            if any(run_label(r)==name for r in record['runs']): continue
            number = 1 + sum(a['method']==name for a in record['attempts'])
            run_root = folder/f'{name}-attempt-{number}'; run_root.mkdir()
            config = dict(record['config'], compression=name, **{'llz-p':0,'topk-k':124000,'output-dir':run_root.as_posix()})
            record['attempts'].append(dict(method=name, output=str(run_root)))
            write_json(folder/'comparison.json', record)
            print(f'Starting {name}: 15 rounds, seed=42', flush=True)
            started = perf_counter(); code = launch(config, root=root, app_dir=stage, log_path=run_root/'terminal.log')
            elapsed = perf_counter()-started
            if code: raise RuntimeError(f'Flower failed ({code}): {run_root / "terminal.log"}')
            run = collect(run_root, config); check_control(reference, run)
            if name=='topk-ef':
                timing = [json.loads(x) for x in (Path(run['output'])/'codec_timing.jsonl').read_text().splitlines()]
                if len(timing)!=60 or {(r['round'],r['client']) for r in timing}!={(r,c) for r in range(1,16) for c in range(4)}:
                    raise ValueError('Incomplete Top-k timing records')
                record['topk_timing_medians'] = {k:statistics.median(r[k] for r in timing) for k in ['topk_encode_seconds','topk_diagnostic_seconds','decode_seconds']}
            record['runs'].append(run); record['launch_seconds'][name]=elapsed
            write_json(folder/'comparison.json', record)
            print(f"Completed {name}: selected accuracy={run['best']['validation_accuracy']:.2%}, uploads={run['train_upload_bytes']:,}, elapsed={elapsed:.1f}s", flush=True)
        record['summary']=summarize(record['runs']); write_json(folder/'summary.json',record['summary'])
        report(folder, record); record['status']='completed'
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed', error=str(error))
        write_json(folder/'comparison.json',record); raise
    write_json(folder/'comparison.json',record)
    print(f'Completed five-method comparison: {folder}', flush=True)


if __name__=='__main__': main()
