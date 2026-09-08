"""Paired seed study with an exact replay gate, fixed data, and saved summaries."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import torch

from flower_face.reproducibility import model_hash, source_hash
from flower_face.experiment import atomic_replace
from flower_face.run import default_config, launch


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    atomic_replace(temporary, path)


def snapshot(root, destination):
    """Copy code only, avoiding repeated scans of 202,599 dataset images."""
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root/'pyproject.toml', destination/'pyproject.toml')
    for package in ('flower_face', 'compression'):
        for path in (root/package).rglob('*.py'):
            target = destination/path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    if source_hash(root) != source_hash(destination):
        raise RuntimeError('Source snapshot differs from working source')


def collect(output_root, expected):
    directories = [p for p in output_root.iterdir() if p.is_dir() and (p/'experiment.json').exists()]
    if len(directories) != 1:
        raise RuntimeError('Expected exactly one experiment in this isolated run directory')
    output = directories[0]
    metadata = json.loads((output/'experiment.json').read_text())
    if metadata['status'] != 'completed' or not (output/'metrics.json').exists():
        raise RuntimeError(f'Experiment did not complete; inspect {output_root / "terminal.log"}')
    metrics = json.loads((output/'metrics.json').read_text())
    history = [json.loads(line) for line in (output/'history.jsonl').read_text().splitlines()]
    if metadata['status'] != 'completed' or metrics['test'] is not None:
        raise RuntimeError('Study requires completed experiments with the test skipped')
    if metrics['config'] != expected:
        raise RuntimeError('Saved configuration differs from requested study run')
    rounds = expected['num-server-rounds']
    if [r['round'] for r in history] != list(range(1, rounds+1)):
        raise RuntimeError('Incomplete or unordered round history')
    if any(r['train_clients'] != expected['num-clients'] or r['validation_clients'] != expected['num-clients'] for r in history):
        raise RuntimeError('Incomplete client participation')
    communication = json.loads((output/'communication.json').read_text())
    messages = [json.loads(line) for line in (output/'communication_messages.jsonl').read_text().splitlines()]
    if (len(messages) != rounds*4*expected['num-clients'] or any(r['has_error'] for r in messages)
            or sum(r['serialized_object_bytes'] for r in messages) != metrics['communication']['total']['serialized_object_bytes']
            or communication['totals'] != metrics['communication']
            or history[-1]['cumulative_communication'] != metrics['communication']):
        raise RuntimeError('Communication records do not reconcile')
    best = torch.load(output/'best_model.pt', map_location='cpu', weights_only=True)
    final = torch.load(output/'final_model.pt', map_location='cpu', weights_only=True)
    selected = min(history, key=lambda r: r['validation']['eval_loss'])
    if (best['step'] != selected['round'] or final['step'] != rounds
            or model_hash(best['state_dict']) != selected['model_sha256']
            or model_hash(final['state_dict']) != history[-1]['model_sha256']):
        raise RuntimeError('Model checkpoint/hash mismatch')
    return dict(output=str(output), config=metrics['config'], metadata=metadata,
                best=metrics['best_checkpoint'], final_validation=history[-1]['validation'],
                total_bytes=metrics['communication']['total']['serialized_object_bytes'],
                train_upload_bytes=metrics['communication']['train']['uplink']['serialized_object_bytes'],
                communication=metrics['communication'],
                replay=[{k: row[k] for k in ('round', 'train', 'validation', 'model_sha256')} for row in history])


def check_pair(a, b):
    for key in ('manifest_sha256', 'source_sha256', 'versions', 'platform', 'reproducibility', 'initial_model_sha256', 'selection_rule'):
        if a['metadata'][key] != b['metadata'][key]:
            raise RuntimeError(f'Run pair mismatch: {key}')
    excluded = {'output-dir', 'compression'}
    if {k:v for k,v in a['config'].items() if k not in excluded} != {k:v for k,v in b['config'].items() if k not in excluded}:
        raise RuntimeError('Run pair training settings differ')


def check_replay(a, b):
    check_pair(a, b)
    if a['config']['compression'] != b['config']['compression'] or a['replay'] != b['replay'] or a['best'] != b['best']:
        raise RuntimeError('Exact replay failed: model hashes or learning metrics differ; study stopped')


def summarize(runs, seeds):
    expected = {(seed, method) for seed in seeds for method in ('none', 'qsgd')}
    actual = [(r['config']['seed'], r['config']['compression']) for r in runs]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError('Expected exactly one run per seed and method')
    pairs = []
    for seed in seeds:
        pair = {r['config']['compression']: r for r in runs if r['config']['seed'] == seed}
        if len(pair) != 2:
            raise RuntimeError('Missing method in seed pair')
        base, qsgd = pair['none'], pair['qsgd']
        check_pair(base, qsgd)
        pairs.append(dict(seed=seed, baseline_output=base['output'], qsgd_output=qsgd['output'],
                          selected_accuracy_difference_pp=100*(qsgd['best']['validation_accuracy']-base['best']['validation_accuracy']),
                          selected_loss_difference=qsgd['best']['validation_loss']-base['best']['validation_loss'],
                          upload_reduction_percent=100*(1-qsgd['train_upload_bytes']/base['train_upload_bytes']),
                          total_reduction_percent=100*(1-qsgd['total_bytes']/base['total_bytes'])))

    def stats(values):
        return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values) if len(values)>1 else None)

    methods = {}
    for method in ('none', 'qsgd'):
        selected = [r for r in runs if r['config']['compression']==method]
        methods[method] = dict(n_seeds=len(selected),
            selected_accuracy=stats([r['best']['validation_accuracy'] for r in selected]),
            selected_loss=stats([r['best']['validation_loss'] for r in selected]),
            final_accuracy=stats([r['final_validation']['eval_acc'] for r in selected]),
            total_bytes=stats([r['total_bytes'] for r in selected]),
            train_upload_bytes=stats([r['train_upload_bytes'] for r in selected]))
    return dict(methods=methods, pairs=pairs, paired_accuracy_difference_pp=stats([p['selected_accuracy_difference_pp'] for p in pairs]),
                note='Sample standard deviation across distinct training seeds on a fixed dataset split. No held-out test or significance claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--rounds', type=int, default=300)
    parser.add_argument('--repeat-rounds', type=int, default=3)
    parser.add_argument('--qsgd-levels', type=int, default=127)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if len(args.seeds)<2 or len(set(args.seeds)) != len(args.seeds) or any(s<0 or s>=2**32 for s in args.seeds):
        parser.error('Use at least two distinct seeds in [0, 2**32-1]')
    if min(args.rounds, args.repeat_rounds)<1 or not 1<=args.qsgd_levels<=65535:
        parser.error('Rounds must be positive and levels in [1, 65535]')
    root = Path(__file__).resolve().parents[1]
    base = default_config(root)
    base.update({'evaluate-final-test': False, 'qsgd-levels': args.qsgd_levels})
    output = (args.output_dir or root/'outputs')/('study-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = output/'app'
    snapshot(root, stage)
    manifest_hash = hashlib.sha256(Path(base['manifest']).read_bytes()).hexdigest()
    source = source_hash(stage)
    record = dict(status='running', seeds=args.seeds, rounds=args.rounds, repeat_rounds=args.repeat_rounds,
                  source_sha256=source, manifest_sha256=manifest_hash, replay_checks={}, runs=[], checks=[])
    print(f'Study directory: {output}', flush=True)

    def execute(label, seed, method, rounds):
        run_root = output/label
        run_root.mkdir()
        config = dict(base, seed=seed, compression=method)
        config.update({'num-server-rounds': rounds, 'output-dir': run_root.as_posix()})
        print(f'Starting {label}: seed={seed}, compression={method}, rounds={rounds}', flush=True)
        if launch(config, root=root, app_dir=stage, log_path=run_root/'terminal.log') != 0:
            raise RuntimeError(f'Flower failed: inspect {run_root / "terminal.log"}')
        result = collect(run_root, config)
        if result['metadata']['source_sha256'] != source or result['metadata']['manifest_sha256'] != manifest_hash:
            raise RuntimeError('Source or manifest changed during the study')
        print(f'Completed {label}: best loss={result["best"]["validation_loss"]:.4f}, accuracy={result["best"]["validation_accuracy"]:.1%}', flush=True)
        return result

    write_json(output/'study.json', record)
    try:
        for method in ('none', 'qsgd'):
            first = execute(f'replay-{method}-1', args.seeds[0], method, args.repeat_rounds)
            record['checks'].append(first)
            write_json(output/'study.json', record)
            second = execute(f'replay-{method}-2', args.seeds[0], method, args.repeat_rounds)
            record['checks'].append(second)
            check_replay(first, second)
            record['replay_checks'][method] = 'exact model hashes and learning metrics matched'
            write_json(output/'study.json', record)
            print(f'Exact replay passed: {method}', flush=True)
        for index, seed in enumerate(args.seeds):
            for method in (('none', 'qsgd') if index%2==0 else ('qsgd', 'none')):
                result = execute(f'seed-{seed}-{method}', seed, method, args.rounds)
                record['runs'].append(result)
                write_json(output/'study.json', record)
        record['summary'] = summarize(record['runs'], args.seeds)
        record['status'] = 'completed'
        write_json(output/'summary.json', record['summary'])
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(output/'study.json', record)
        raise
    write_json(output/'study.json', record)
    print(f'Completed study. Summary: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
