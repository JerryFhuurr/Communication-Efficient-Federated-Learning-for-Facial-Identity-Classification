"""Compare SGD weight decay after exact full-length replay of the old baseline."""

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


def normalized_config(config):
    result = dict(config)
    for key, value in (('weight-decay', 0.), ('llz-p', 0), ('llz-window', 128)):
        result.setdefault(key, value)
    return result


def check_control(base, candidate, *, legacy_replay=False):
    a, b = normalized_config(base['config']), normalized_config(candidate['config'])
    excluded = {'output-dir'} if legacy_replay else {'output-dir', 'weight-decay'}
    if {k:v for k,v in a.items() if k not in excluded} != {k:v for k,v in b.items() if k not in excluded}:
        raise RuntimeError('Weight-decay comparison contains a training confound')
    if b['compression'] != 'none' or b['evaluate-final-test']:
        raise RuntimeError('Require uncompressed training with test unused')
    keys = ['manifest_sha256', 'initial_model_sha256', 'versions', 'python', 'platform',
            'reproducibility', 'selection_rule']
    if not legacy_replay:
        keys += ['source_sha256', 'communication_measurement']
    for key in keys:
        if base['metadata'][key] != candidate['metadata'][key]:
            raise RuntimeError(f'Weight-decay control mismatch: {key}')
    if legacy_replay:
        if a['weight-decay'] != 0 or b['weight-decay'] != 0:
            raise RuntimeError('Legacy replay must use zero weight decay')
        if base['replay'] != candidate['replay'] or base['best'] != candidate['best']:
            raise RuntimeError('Zero-decay model/metric/checkpoint replay differs from legacy baseline')


def summarize(runs, seeds, decays):
    pairs = [(r['config']['seed'], r['config']['weight-decay']) for r in runs]
    if len(pairs) != len(seeds)*len(decays) or set(pairs) != {(s,w) for s in seeds for w in decays}:
        raise RuntimeError('Incomplete or duplicate seed/decay results')
    rows = []
    for decay in decays:
        selected = [r for r in runs if r['config']['weight-decay']==decay]
        losses = [r['best']['validation_loss'] for r in selected]
        accuracies = [r['best']['validation_accuracy'] for r in selected]
        if not all(math.isfinite(v) for v in losses+accuracies):
            raise RuntimeError('Nonfinite comparison metric')
        rows.append(dict(weight_decay=decay, mean_selected_loss=statistics.mean(losses),
            sample_sd_selected_loss=statistics.stdev(losses), mean_selected_accuracy=statistics.mean(accuracies),
            sample_sd_selected_accuracy=statistics.stdev(accuracies),
            mean_final_train_accuracy=statistics.mean(r['checkpoint_evaluation']['final']['train']['accuracy'] for r in selected)))
    winner = min(rows, key=lambda r:(r['mean_selected_loss'], r['weight_decay']))
    return dict(decays=rows, preferred_weight_decay=winner['weight_decay'],
        selection_rule='Lowest mean validation-selected loss across seeds; smaller decay wins exact ties.',
        interpretation='Exploratory tuning on a repeatedly used 40-image validation split; test unused.')


def curves(runs):
    return [dict(seed=r['config']['seed'], weight_decay=r['config']['weight-decay'], output=r['output'],
        best=r['best'], checkpoint_evaluation=r['checkpoint_evaluation'],
        history=[dict(step=h['round'], optimization_loss=h['train']['train_loss'],
                      validation_loss=h['validation']['eval_loss'], validation_accuracy=h['validation']['eval_acc'])
                 for h in r['replay']]) for r in runs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT/'outputs/lr-sweep-20260909T093522322142Z/sweep.json')
    parser.add_argument('--decays', type=float, nargs='+', default=[0., .0001, .001])
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if len(args.decays)<2 or len(set(args.decays)) != len(args.decays) or 0 not in args.decays or any(not math.isfinite(x) or x<0 for x in args.decays):
        parser.error('Use distinct finite nonnegative decays including zero and a positive candidate')
    if args.resume:
        output = args.resume.resolve()
        record = read(output/'sweep.json')
        protocol = record['protocol']
        reference_path = Path(protocol['reference'])
        if digest(__file__) != protocol['runner_sha256'] or digest(reference_path) != protocol['reference_sha256']:
            raise RuntimeError('Runner or reference changed; cannot resume this sweep')
    else:
        reference_path = args.reference.resolve()
        output = ROOT/'outputs'/('wd-sweep-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        output.mkdir(parents=True, exist_ok=False)
        record = dict(status='preparing', runs=[], attempts=[], replay_checks={})
    print(f'Weight-decay sweep directory: {output}', flush=True)
    reference = read(reference_path)
    if reference['status'] != 'completed':
        raise RuntimeError('Learning-rate reference is incomplete')
    legacy = [collect(Path(r['output']).parent, r['config']) for r in reference['runs'] if r['config']['learning-rate']==.1]
    seeds = sorted(r['config']['seed'] for r in legacy)
    if seeds != [42,43,44]:
        raise RuntimeError('Expected legacy rate-0.1 runs for seeds 42, 43 and 44')
    stage = output/'app'
    if not args.resume:
        snapshot(ROOT, stage)
        base_config = default_config(ROOT)
        base_config.update({'learning-rate': .1, 'weight-decay': 0., 'num-server-rounds': 300,
                            'compression': 'none', 'evaluate-final-test': False})
        protocol = dict(reference=str(reference_path), reference_sha256=digest(reference_path),
            runner_sha256=digest(__file__), source_sha256=source_hash(stage),
            legacy_source_sha256=legacy[0]['metadata']['source_sha256'],
            manifest_sha256=digest(base_config['manifest']), config=base_config,
            seeds=seeds, decays=sorted(args.decays), rounds=300, learning_rate=.1,
            selection_rule='Lowest mean validation-selected loss across seeds; smaller decay wins exact ties.',
            regularization='PyTorch SGD coupled L2 on all parameters, including biases; reported loss is cross-entropy.',
            test_evaluated=False)
        record['protocol'] = protocol
    if source_hash(stage) != protocol['source_sha256'] or source_hash(ROOT) != protocol['source_sha256']:
        raise RuntimeError('Training source changed')
    if digest(protocol['config']['manifest']) != protocol['manifest_sha256']:
        raise RuntimeError('Dataset manifest changed')
    for r in legacy:
        if r['metadata']['source_sha256'] != protocol['legacy_source_sha256'] or r['metadata']['manifest_sha256'] != protocol['manifest_sha256']:
            raise RuntimeError('Legacy baseline fingerprint differs')
    for saved in record['runs']:
        fresh = collect(Path(saved['output']).parent, saved['config'])
        if fresh['replay'] != saved['replay'] or fresh['best'] != saved['best']:
            raise RuntimeError('Completed artifacts changed')
        if saved['config']['weight-decay']==0:
            check_control(next(r for r in legacy if r['config']['seed']==saved['config']['seed']), fresh, legacy_replay=True)
        else:
            check_control(next(r for r in record['runs'] if r['config']['seed']==saved['config']['seed'] and r['config']['weight-decay']==0), fresh)
    record['status'] = 'running'
    write_json(output/'sweep.json', record)
    try:
        # Finish all full-length zero-decay replays before any positive-decay run.
        plan = [(seed, 0.) for seed in seeds]
        positive = [v for v in protocol['decays'] if v>0]
        for index, seed in enumerate(seeds):
            plan.extend((seed, decay) for decay in (positive if index%2==0 else positive[::-1]))
        for seed, decay in plan:
            if any(r['config']['seed']==seed and r['config']['weight-decay']==decay for r in record['runs']):
                continue
            if decay>0 and set(record['replay_checks']) != {str(s) for s in seeds}:
                raise RuntimeError('All zero-decay replay checks must pass before regularization runs')
            attempt_index = 1 + sum(a['seed']==seed and a['weight_decay']==decay for a in record['attempts'])
            run_root = output/f'seed-{seed}-wd-{decay:g}-attempt-{attempt_index}'
            run_root.mkdir()
            config = dict(protocol['config'])
            config.update({'seed':seed, 'weight-decay':decay, 'output-dir':run_root.as_posix()})
            attempt = dict(seed=seed, weight_decay=decay, output=str(run_root), status='running')
            record['attempts'].append(attempt)
            write_json(output/'sweep.json', record)
            print(f'Starting seed={seed}, weight-decay={decay:g}, lr=0.1, rounds=300', flush=True)
            try:
                if source_hash(stage) != protocol['source_sha256'] or digest(config['manifest']) != protocol['manifest_sha256']:
                    raise RuntimeError('Application or manifest changed before launch')
                if launch(config, root=ROOT, app_dir=stage, log_path=run_root/'terminal.log') != 0:
                    raise RuntimeError(f'Flower failed: {run_root / "terminal.log"}')
                result = collect(run_root, config)
                if result['metadata']['source_sha256'] != protocol['source_sha256']:
                    raise RuntimeError('Run source does not match the frozen app')
                base = next(r for r in (legacy if decay==0 else record['runs'])
                            if r['config']['seed']==seed and r['config'].get('weight-decay',0)==0)
                check_control(base, result, legacy_replay=decay==0)
                add_evaluation(result)
                record['runs'].append(result)
                attempt['status'] = 'completed'
                if decay==0:
                    record['replay_checks'][str(seed)] = 'All 300 model hashes, metrics and selected checkpoint match legacy exactly'
                    print(f'Exact full-length zero-decay replay passed: seed {seed}', flush=True)
            except BaseException as error:
                attempt.update(status='failed', error=str(error))
                raise
            write_json(output/'sweep.json', record)
            write_json(output/'curves.json', dict(status='running', curves=curves(record['runs'])))
            print(f'Completed seed={seed}, decay={decay:g}: selected round={result["best"]["step"]}, '
                  f'loss={result["best"]["validation_loss"]:.6f}, accuracy={result["best"]["validation_accuracy"]:.1%}', flush=True)
        record['summary'] = summarize(record['runs'], seeds, protocol['decays'])
        record['status'] = 'completed'
        write_json(output/'sweep.json', record)
        write_json(output/'curves.json', dict(status='completed', curves=curves(record['runs'])))
        write_json(output/'summary.json', record['summary'])
        print(f'Completed weight-decay sweep: {output / "summary.json"}', flush=True)
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(output/'sweep.json', record)
        raise


if __name__=='__main__':
    main()
