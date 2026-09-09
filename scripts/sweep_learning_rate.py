"""Controlled uncompressed FedAvg learning-rate comparison on a frozen study app."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flower_face.reproducibility import source_hash
from flower_face.run import launch
from flower_face.study import collect, snapshot, write_json
from flower_face.task import seed_everything
from scripts.diagnose_training import checkpoint_metrics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_control(reference, candidate):
    """Learning rate/output path are the only allowed within-seed differences."""
    excluded = {'learning-rate', 'output-dir'}
    if {k: v for k, v in reference['config'].items() if k not in excluded} != {
            k: v for k, v in candidate['config'].items() if k not in excluded}:
        raise RuntimeError('Training controls changed beyond learning rate')
    if candidate['config']['compression'] != 'none' or candidate['config']['evaluate-final-test']:
        raise RuntimeError('Sweep requires uncompressed training and an unused test set')
    for key in ('source_sha256', 'manifest_sha256', 'initial_model_sha256', 'versions',
                'python', 'platform', 'reproducibility', 'selection_rule', 'communication_measurement'):
        if reference['metadata'][key] != candidate['metadata'][key]:
            raise RuntimeError(f'Controlled comparison mismatch: {key}')


def summarize(runs, seeds, rates):
    keys = [(r['config']['seed'], r['config']['learning-rate']) for r in runs]
    if len(keys) != len(seeds)*len(rates) or set(keys) != {(s, lr) for s in seeds for lr in rates}:
        raise RuntimeError('Expected exactly one completed run per seed and learning rate')
    rows = []
    for lr in rates:
        selected = [r for r in runs if r['config']['learning-rate'] == lr]
        loss = [r['best']['validation_loss'] for r in selected]
        accuracy = [r['best']['validation_accuracy'] for r in selected]
        if not all(math.isfinite(v) for v in loss + accuracy):
            raise RuntimeError('Nonfinite selection metric')
        rows.append(dict(learning_rate=lr, mean_selected_loss=statistics.mean(loss),
            sample_sd_selected_loss=statistics.stdev(loss),
            mean_selected_accuracy=statistics.mean(accuracy), sample_sd_selected_accuracy=statistics.stdev(accuracy),
            mean_final_train_accuracy=statistics.mean(r['checkpoint_evaluation']['final']['train']['accuracy'] for r in selected)))
    winner = min(rows, key=lambda r: (r['mean_selected_loss'], r['learning_rate']))
    return dict(rates=rows, preferred_learning_rate=winner['learning_rate'],
        selection_rule='Lowest mean validation-selected loss across all seeds; smaller rate wins exact ties.',
        interpretation='Exploratory tuning on one repeatedly used validation split; no held-out test or significance claim.')


def add_evaluation(run):
    seed_everything(run['config']['seed'])
    run['checkpoint_evaluation'] = {name: checkpoint_metrics(Path(run['output'])/f'{name}_model.pt', run['config'])
                                    for name in ('best', 'final')}
    best = run['checkpoint_evaluation']['best']['validation']
    if (abs(best['loss'] - run['best']['validation_loss']) > 1e-5
            or abs(best['accuracy'] - run['best']['validation_accuracy']) > 1e-12):
        raise RuntimeError('Checkpoint re-evaluation differs from saved validation metrics')


def curves(runs):
    return [dict(seed=r['config']['seed'], learning_rate=r['config']['learning-rate'],
        best=r['best'], output=r['output'], checkpoint_evaluation=r['checkpoint_evaluation'],
        history=[dict(step=h['round'], optimization_loss=h['train']['train_loss'],
                      validation_loss=h['validation']['eval_loss'], validation_accuracy=h['validation']['eval_acc'])
                 for h in r['replay']]) for r in runs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT/'outputs/study-20260908T114723972670Z/study.json')
    parser.add_argument('--rates', type=float, nargs='+', default=[0.01, 0.03, 0.1])
    parser.add_argument('--resume', type=Path, help='Resume a failed/interrupted sweep without repeating completed runs')
    args = parser.parse_args()
    if len(args.rates)<2 or len(set(args.rates)) != len(args.rates) or any(not math.isfinite(r) or r <= 0 for r in args.rates):
        parser.error('Use at least two distinct finite positive learning rates')
    if args.resume:
        output = args.resume.resolve()
        record = read(output/'sweep.json')
        protocol = record['protocol']
        if digest(__file__) != protocol['runner_sha256']:
            raise RuntimeError('Runner changed since the sweep began')
        reference_path = Path(protocol['reference'])
    else:
        reference_path = args.reference.resolve()
        output = ROOT/'outputs'/('lr-sweep-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        output.mkdir(parents=True, exist_ok=False)
        record = dict(status='preparing', runs=[], attempts=[])
    print(f'Sweep directory: {output}', flush=True)
    reference = read(reference_path)
    if reference['status'] != 'completed':
        raise RuntimeError('Reference study is incomplete')
    baselines = [collect(Path(r['output']).parent, r['config']) for r in reference['runs'] if r['config']['compression']=='none']
    seeds = sorted(r['config']['seed'] for r in baselines)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise RuntimeError('Reference needs at least two unique baseline seeds')
    reference_rate = baselines[0]['config']['learning-rate']
    if any(r['config']['learning-rate'] != reference_rate for r in baselines):
        raise RuntimeError('Reference learning rates differ')
    stage = output/'app'
    if not args.resume:
        rates = sorted(args.rates)
        if reference_rate not in rates:
            raise RuntimeError('Rates must include the existing baseline learning rate')
        snapshot(reference_path.parent/'app', stage)
        protocol = dict(reference=str(reference_path), reference_sha256=digest(reference_path),
            runner_sha256=digest(__file__), seeds=seeds, rates=rates, reference_rate=reference_rate,
            source_sha256=source_hash(stage), manifest_sha256=baselines[0]['metadata']['manifest_sha256'],
            rounds=baselines[0]['config']['num-server-rounds'],
            selection_rule='Lowest mean validation-selected loss across all seeds; smaller rate wins exact ties.',
            test_evaluated=False, model_defaults_changed=False)
        record['protocol'] = protocol
    else:
        rates = protocol['rates']
        if digest(reference_path) != protocol['reference_sha256']:
            raise RuntimeError('Reference study changed')
    if source_hash(stage) != protocol['source_sha256'] or source_hash(reference_path.parent/'app') != protocol['source_sha256']:
        raise RuntimeError('Frozen source differs from reference')
    if digest(ROOT/'flower_face/task.py') != digest(stage/'flower_face/task.py'):
        raise RuntimeError('Current model/evaluation code differs from frozen reference')
    for r in baselines:
        if (r['metadata']['source_sha256'] != protocol['source_sha256']
                or digest(r['config']['manifest']) != protocol['manifest_sha256']):
            raise RuntimeError('Baseline source or dataset fingerprint mismatch')
        check_control(baselines[0] if r['config']['seed']==seeds[0] else r, r)
    # Re-audit completed runs on resume before accepting their recorded results.
    for saved in record['runs']:
        fresh = collect(Path(saved['output']).parent, saved['config'])
        if fresh['replay'] != saved['replay'] or fresh['best'] != saved['best']:
            raise RuntimeError('Completed run artifacts changed')
        check_control(next(r for r in baselines if r['config']['seed']==saved['config']['seed']), fresh)
    record['status'] = 'running'
    write_json(output/'sweep.json', record)
    try:
        if not record['runs']:
            for base in baselines:
                add_evaluation(base)
                record['runs'].append(base)
            write_json(output/'sweep.json', record)
        plan = []
        for index, seed in enumerate(seeds):
            candidate_rates = [r for r in rates if r != reference_rate]
            if index % 2:
                candidate_rates.reverse()
            plan.extend((seed, lr) for lr in candidate_rates)
        for seed, lr in plan:
            if any(r['config']['seed']==seed and r['config']['learning-rate']==lr for r in record['runs']):
                continue
            base = next(r for r in baselines if r['config']['seed']==seed)
            attempt_number = 1 + sum(a['seed']==seed and a['learning_rate']==lr for a in record['attempts'])
            run_root = output/f'seed-{seed}-lr-{lr:g}-attempt-{attempt_number}'
            run_root.mkdir()
            config = dict(base['config'])
            config.update({'learning-rate': lr, 'output-dir': run_root.as_posix()})
            attempt = dict(seed=seed, learning_rate=lr, output=str(run_root), status='running')
            record['attempts'].append(attempt)
            write_json(output/'sweep.json', record)
            print(f'Starting seed={seed}, lr={lr:g}, rounds={config["num-server-rounds"]}', flush=True)
            try:
                if source_hash(stage) != protocol['source_sha256'] or digest(config['manifest']) != protocol['manifest_sha256']:
                    raise RuntimeError('Source or manifest changed before launch')
                if launch(config, root=ROOT, app_dir=stage, log_path=run_root/'terminal.log') != 0:
                    raise RuntimeError(f'Flower failed: {run_root / "terminal.log"}')
                result = collect(run_root, config)
                check_control(base, result)
                add_evaluation(result)
                record['runs'].append(result)
                attempt['status'] = 'completed'
            except BaseException as error:
                attempt.update(status='failed', error=str(error))
                raise
            write_json(output/'sweep.json', record)
            write_json(output/'curves.json', dict(status='running', curves=curves(record['runs'])))
            print(f'Completed seed={seed}, lr={lr:g}: selected round={result["best"]["step"]}, '
                  f'validation loss={result["best"]["validation_loss"]:.6f}, '
                  f'accuracy={result["best"]["validation_accuracy"]:.1%}', flush=True)
        record['summary'] = summarize(record['runs'], seeds, rates)
        record['status'] = 'completed'
        write_json(output/'curves.json', dict(status='completed', curves=curves(record['runs'])))
        write_json(output/'sweep.json', record)
        write_json(output/'summary.json', record['summary'])
        print(f'Completed sweep: {output / "summary.json"}', flush=True)
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(output/'sweep.json', record)
        raise


if __name__ == '__main__':
    main()
