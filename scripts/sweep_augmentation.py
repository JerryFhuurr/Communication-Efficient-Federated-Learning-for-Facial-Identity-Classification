"""Compare training-only horizontal flips after exact no-augmentation replay."""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flower_face.reproducibility import source_hash
from flower_face.run import default_config, launch
from flower_face.study import collect, snapshot, write_json
from scripts.sweep_learning_rate import add_evaluation, digest, read


def normalized(config):
    result = dict(config)
    result.setdefault('augmentation', 'none')
    return result


def check_control(base, candidate, *, legacy_replay=False):
    a, b = normalized(base['config']), normalized(candidate['config'])
    excluded = {'output-dir'} if legacy_replay else {'output-dir', 'augmentation'}
    if {k:v for k,v in a.items() if k not in excluded} != {k:v for k,v in b.items() if k not in excluded}:
        raise RuntimeError('Augmentation comparison contains a training confound')
    if b['compression'] != 'none' or b['evaluate-final-test']:
        raise RuntimeError('Require uncompressed training with test unused')
    keys = ['manifest_sha256', 'initial_model_sha256', 'versions', 'python', 'platform',
            'reproducibility', 'selection_rule']
    if not legacy_replay:
        keys += ['source_sha256', 'communication_measurement']
    for key in keys:
        if base['metadata'][key] != candidate['metadata'][key]:
            raise RuntimeError(f'Augmentation control mismatch: {key}')
    if legacy_replay:
        if a['augmentation'] != 'none' or b['augmentation'] != 'none':
            raise RuntimeError('Compatibility replay must disable augmentation')
        if base['replay'] != candidate['replay'] or base['best'] != candidate['best']:
            raise RuntimeError('No-augmentation replay differs from its legacy baseline')


def summarize(runs, seeds, methods):
    observed = [(r['config']['seed'], r['config']['augmentation']) for r in runs]
    if len(observed) != len(seeds)*len(methods) or set(observed) != {(s,m) for s in seeds for m in methods}:
        raise RuntimeError('Incomplete or duplicate seed/augmentation results')
    rows = []
    for method in methods:
        selected = [r for r in runs if r['config']['augmentation']==method]
        losses = [r['best']['validation_loss'] for r in selected]
        accuracies = [r['best']['validation_accuracy'] for r in selected]
        if not all(math.isfinite(v) for v in losses+accuracies):
            raise RuntimeError('Nonfinite comparison metric')
        rows.append(dict(augmentation=method, mean_selected_loss=statistics.mean(losses),
            sample_sd_selected_loss=statistics.stdev(losses), mean_selected_accuracy=statistics.mean(accuracies),
            sample_sd_selected_accuracy=statistics.stdev(accuracies),
            mean_final_train_accuracy=statistics.mean(r['checkpoint_evaluation']['final']['train']['accuracy'] for r in selected)))
    order = {method:index for index,method in enumerate(methods)}
    winner = min(rows, key=lambda r:(r['mean_selected_loss'], order[r['augmentation']]))
    return dict(augmentations=rows, preferred_augmentation=winner['augmentation'],
        selection_rule='Lowest mean validation-selected loss across seeds; earlier declared method wins exact ties.',
        interpretation='Exploratory tuning on a repeatedly used 40-image validation split; test images unused.')


def curves(runs):
    return [dict(seed=r['config']['seed'], augmentation=r['config']['augmentation'], output=r['output'],
        best=r['best'], checkpoint_evaluation=r['checkpoint_evaluation'],
        history=[dict(step=h['round'], optimization_loss=h['train']['train_loss'],
                      validation_loss=h['validation']['eval_loss'], validation_accuracy=h['validation']['eval_acc'])
                 for h in r['replay']]) for r in runs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT/'outputs/wd-sweep-20260909T100745974156Z/sweep.json')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.resume:
        output = args.resume.resolve()
        record = read(output/'sweep.json')
        protocol = record['protocol']
        reference_path = Path(protocol['reference'])
        if digest(__file__) != protocol['runner_sha256'] or digest(reference_path) != protocol['reference_sha256']:
            raise RuntimeError('Runner or reference changed; cannot resume')
    else:
        reference_path = args.reference.resolve()
        output = ROOT/'outputs'/('augmentation-sweep-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        output.mkdir(parents=True, exist_ok=False)
        record = dict(status='preparing', runs=[], attempts=[], replay_checks={})
    print(f'Augmentation sweep directory: {output}', flush=True)
    reference = read(reference_path)
    if reference['status'] != 'completed':
        raise RuntimeError('Weight-decay reference is incomplete')
    legacy = [collect(Path(r['output']).parent, r['config']) for r in reference['runs']
              if r['config']['learning-rate']==.1 and r['config']['weight-decay']==0]
    seeds = sorted(r['config']['seed'] for r in legacy)
    if seeds != [42,43,44]:
        raise RuntimeError('Expected zero-decay rate-0.1 references for seeds 42, 43 and 44')
    stage = output/'app'
    if not args.resume:
        snapshot(ROOT, stage)
        config = default_config(ROOT)
        config.update({'learning-rate':.1, 'weight-decay':0., 'augmentation':'none',
                       'num-server-rounds':300, 'compression':'none', 'evaluate-final-test':False})
        protocol = dict(reference=str(reference_path), reference_sha256=digest(reference_path),
            runner_sha256=digest(__file__), source_sha256=source_hash(stage),
            legacy_source_sha256=legacy[0]['metadata']['source_sha256'], manifest_sha256=digest(config['manifest']),
            config=config, seeds=seeds, augmentations=['none','horizontal-flip'], rounds=300,
            learning_rate=.1, weight_decay=0.,
            augmentation='RandomHorizontalFlip(p=0.5) after resize, training split only; validation/test and checkpoint evaluation unaugmented.',
            selection_rule='Lowest mean validation-selected loss across seeds; none wins exact ties.',
            test_evaluated=False)
        record['protocol'] = protocol
    if source_hash(ROOT) != protocol['source_sha256'] or source_hash(stage) != protocol['source_sha256']:
        raise RuntimeError('Training source changed')
    if digest(protocol['config']['manifest']) != protocol['manifest_sha256']:
        raise RuntimeError('Dataset manifest changed')
    for saved in record['runs']:
        fresh = collect(Path(saved['output']).parent, saved['config'])
        if fresh['replay'] != saved['replay'] or fresh['best'] != saved['best']:
            raise RuntimeError('Completed run artifacts changed')
        base = next(r for r in legacy if r['config']['seed']==saved['config']['seed'])
        if saved['config']['augmentation']=='none':
            check_control(base, fresh, legacy_replay=True)
        else:
            current = next(r for r in record['runs'] if r['config']['seed']==saved['config']['seed'] and r['config']['augmentation']=='none')
            check_control(current, fresh)
    record['status'] = 'running'
    write_json(output/'sweep.json', record)
    try:
        plan = [(seed,'none') for seed in seeds] + [(seed,'horizontal-flip') for seed in seeds]
        for seed, augmentation in plan:
            if any(r['config']['seed']==seed and r['config']['augmentation']==augmentation for r in record['runs']):
                continue
            if augmentation!='none' and set(record['replay_checks']) != {str(s) for s in seeds}:
                raise RuntimeError('All no-augmentation compatibility replays must pass first')
            attempt_index = 1 + sum(a['seed']==seed and a['augmentation']==augmentation for a in record['attempts'])
            run_root = output/f'seed-{seed}-augmentation-{augmentation}-attempt-{attempt_index}'
            run_root.mkdir()
            config = dict(protocol['config'], seed=seed, augmentation=augmentation, **{'output-dir':run_root.as_posix()})
            attempt = dict(seed=seed, augmentation=augmentation, output=str(run_root), status='running')
            record['attempts'].append(attempt)
            write_json(output/'sweep.json', record)
            print(f'Starting seed={seed}, augmentation={augmentation}, rounds=300', flush=True)
            try:
                if source_hash(stage)!=protocol['source_sha256'] or digest(config['manifest'])!=protocol['manifest_sha256']:
                    raise RuntimeError('Source or manifest changed before launch')
                if launch(config, root=ROOT, app_dir=stage, log_path=run_root/'terminal.log') != 0:
                    raise RuntimeError(f'Flower failed: {run_root / "terminal.log"}')
                result = collect(run_root, config)
                if result['metadata']['source_sha256'] != protocol['source_sha256']:
                    raise RuntimeError('Run source differs from frozen application')
                if augmentation=='none':
                    check_control(next(r for r in legacy if r['config']['seed']==seed), result, legacy_replay=True)
                    record['replay_checks'][str(seed)] = 'All 300 model hashes, metrics and selected checkpoint match legacy exactly'
                    print(f'Exact no-augmentation replay passed: seed {seed}', flush=True)
                else:
                    check_control(next(r for r in record['runs'] if r['config']['seed']==seed and r['config']['augmentation']=='none'), result)
                add_evaluation(result)
                record['runs'].append(result)
                attempt['status'] = 'completed'
            except BaseException as error:
                attempt.update(status='failed', error=str(error))
                raise
            write_json(output/'sweep.json', record)
            write_json(output/'curves.json', dict(status='running', curves=curves(record['runs'])))
            print(f'Completed seed={seed}, augmentation={augmentation}: selected round={result["best"]["step"]}, '
                  f'loss={result["best"]["validation_loss"]:.6f}, accuracy={result["best"]["validation_accuracy"]:.1%}', flush=True)
        record['summary'] = summarize(record['runs'], seeds, protocol['augmentations'])
        record['status'] = 'completed'
        write_json(output/'sweep.json', record)
        write_json(output/'curves.json', dict(status='completed', curves=curves(record['runs'])))
        write_json(output/'summary.json', record['summary'])
        print(f'Completed augmentation sweep: {output / "summary.json"}', flush=True)
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed', error=str(error))
        write_json(output/'sweep.json', record)
        raise


if __name__=='__main__':
    main()
