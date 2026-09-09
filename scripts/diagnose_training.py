"""Audit saved federated curves and run equal-data-exposure centralized references.

Run with the project's Python environment. No model or dataset changes; no test
images are loaded. Outputs include machine-readable curves and checkpoints.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from flower_face.experiment import Experiment
from flower_face.reproducibility import model_hash, source_hash
from flower_face.study import collect, write_json
from flower_face.task import Net, load_data, read_manifest, seed_everything, test, train


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_metrics(path, config):
    """Evaluate train/validation only, with the same preprocessing and model."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = Net(config['num-classes'])
    model.load_state_dict(saved['state_dict'])
    values = {}
    for split in ('train', 'validation'):
        dataset = load_data(config, split=split, apply_augmentation=False).dataset
        loader = DataLoader(dataset, batch_size=config['batch-size'], shuffle=False)
        loss, accuracy = test(model, loader)
        values[split] = dict(loss=loss, accuracy=accuracy)
    return dict(step=saved['step'], model_sha256=model_hash(model.state_dict()), **values)


def central_worker(request_path):
    request = read(request_path)
    config = request['config']
    seed_everything(config['seed'])
    manifest = read_manifest(config['manifest'], config['num-classes'], config['num-clients'])
    model = Net(config['num-classes'])
    initial_hash = model_hash(model.state_dict())
    if initial_hash != request['initial_model_sha256']:
        raise RuntimeError('Centralized initialization differs from the federated reference')
    loader = load_data(config, split='train')
    evaluation = load_data(config, split='train', apply_augmentation=False)
    validation = load_data(config, split='validation')
    experiment = Experiment(request['output'], 'centralized', config, manifest, 'epoch')
    experiment.metadata.update(initial_model_sha256=initial_hash,
        diagnostic_script_sha256=file_hash(__file__), requested_epochs=request['epochs'],
        budget_basis='Equal total training-example presentations; not equal optimizer trajectories')
    experiment.write_json('experiment.json', experiment.metadata)
    history = []
    for epoch in range(1, request['epochs'] + 1):
        optimization_loss = train(model, loader, epochs=1, lr=config['learning-rate'],
                                  weight_decay=config.get('weight-decay', 0.0))
        train_loss, train_acc = test(model, evaluation)
        val_loss, val_acc = test(model, validation)
        row = dict(step=epoch, optimization_loss=optimization_loss,
            train_loss=train_loss, train_accuracy=train_acc,
            validation_loss=val_loss, validation_accuracy=val_acc,
            model_sha256=model_hash(model.state_dict()))
        if not all(math.isfinite(row[k]) for k in ('optimization_loss', 'train_loss', 'validation_loss')):
            raise RuntimeError('Nonfinite centralized loss')
        experiment.consider_best(model.state_dict(), epoch, val_loss, val_acc)
        history.append(row)
        experiment.log_step(row)
        if epoch == 1 or epoch % 25 == 0 or epoch == request['epochs']:
            print(f'Seed {config["seed"]} epoch {epoch}/{request["epochs"]}: '
                  f'train={train_acc:.1%}, validation={val_acc:.1%}, loss={val_loss:.4f}', flush=True)
    experiment.save_checkpoint('final_model.pt', model.state_dict(), request['epochs'], history[-1])
    experiment.complete(dict(config=config, history=history, test_evaluated=False,
        completed_epochs=len(history), training_presentations=len(loader.dataset)*len(history),
        optimizer_steps=len(loader)*len(history)))
    write_json(Path(request['output'])/'result.json', dict(output=str(experiment.output)))


def audit_references(study_path, llz_path, destination):
    study, llz = read(study_path), read(llz_path)
    if study['status'] != 'completed' or llz['status'] != 'completed':
        raise RuntimeError('Both reference studies must be complete')
    if set(study['seeds']) != {42, 43, 44} or study['rounds'] != 300:
        raise RuntimeError('Expected the completed 300-round, three-seed reference')
    if file_hash(ROOT/'flower_face/task.py') != file_hash(study_path.parent/'app/flower_face/task.py'):
        raise RuntimeError('Model, preprocessing or training code changed since the saved FedAvg study')
    runs, curves = [], []
    for saved in study['runs']:
        run = collect(Path(saved['output']).parent, saved['config'])
        if run['replay'] != saved['replay'] or run['best'] != saved['best']:
            raise RuntimeError('Saved study differs from its experiment artifacts')
        seed, method = run['config']['seed'], run['config']['compression']
        if method == 'qsgd':
            paired_path = Path(next(r['comparison'] for r in llz['results'] if r['seed'] == seed))
            paired = read(paired_path)
            if paired['status'] != 'completed' or not paired['summary']['exact_replay']:
                raise RuntimeError('LLZ reference did not pass exact replay')
            if run['replay'] != paired['runs'][0]['replay']:
                raise RuntimeError('Older QSGD learning curves differ from the LLZ study reference')
        curve = dict(seed=seed, method=method, output=run['output'], best=run['best'],
            history=[dict(step=r['round'], optimization_loss=r['train']['train_loss'],
                validation_loss=r['validation']['eval_loss'], validation_accuracy=r['validation']['eval_acc'])
                for r in run['replay']])
        curves.append(curve)
        runs.append(run)
    if {(r['config']['seed'], r['config']['compression']) for r in runs} != {
            (seed, method) for seed in (42, 43, 44) for method in ('none', 'qsgd')} or len(runs) != 6:
        raise RuntimeError('Missing or duplicate reference run')
    baseline_config = {k: v for k, v in runs[0]['config'].items() if k not in ('seed', 'compression', 'output-dir')}
    for run in runs:
        if {k: v for k, v in run['config'].items() if k not in ('seed', 'compression', 'output-dir')} != baseline_config:
            raise RuntimeError('Reference training configurations differ')
        if run['metadata']['manifest_sha256'] != file_hash(run['config']['manifest']):
            raise RuntimeError('Dataset manifest changed')
    write_json(destination/'curves.json', dict(status='federated_only', curves=curves))
    return runs, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT/'outputs/study-20260908T114723972670Z/study.json')
    parser.add_argument('--llz-summary', type=Path,
        default=ROOT/'outputs/llz-three-seeds-20260908T132726021518Z/summary.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--central-worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.central_worker:
        central_worker(args.central_worker)
        return
    output = (args.output_dir or ROOT/'outputs'/('training-diagnosis-'+
              datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f'Diagnosis directory: {output}', flush=True)
    source = source_hash(ROOT)
    record = dict(status='running', source_sha256=source,
        script_sha256=file_hash(__file__), references={str(p): file_hash(p) for p in (args.study, args.llz_summary)},
        test_evaluated=False)
    write_json(output/'diagnosis.json', record)
    jobs, logs = [], []
    try:
        runs, curves = audit_references(args.study, args.llz_summary, output)
        print('Audited six federated runs. Older QSGD curves exactly match the LLZ study.', flush=True)
        for seed in (42, 43, 44):
            reference = next(r for r in runs if r['config']['seed'] == seed and r['config']['compression'] == 'none')
            config = dict(reference['config'])
            run_root = output/f'seed-{seed}'
            run_root.mkdir()
            config.update({'output-dir': str(run_root), 'compression': 'none', 'evaluate-final-test': False})
            request = dict(config=config, epochs=config['num-server-rounds']*config['local-epochs'],
                initial_model_sha256=reference['metadata']['initial_model_sha256'], output=str(run_root))
            write_json(run_root/'request.json', request)
            log = (run_root/'terminal.log').open('w', encoding='utf-8')
            logs.append(log)
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--central-worker', str(run_root/'request.json')],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            jobs.append((seed, child, run_root))
        while any(p.poll() is None for _, p, _ in jobs):
            for seed, child, _ in jobs:
                if child.poll() not in (None, 0):
                    raise RuntimeError(f'Centralized seed {seed} failed; inspect its terminal.log')
            progress = []
            for seed, child, run_root in jobs:
                paths = list(run_root.glob('centralized-*/history.jsonl'))
                count = len(paths[0].read_text().splitlines()) if paths else 0
                progress.append(f'{seed}: {count}/300')
            print('Centralized epochs — '+', '.join(progress), flush=True)
            time.sleep(15)
        if any(child.returncode for _, child, _ in jobs):
            raise RuntimeError('A centralized worker failed')
        for log in logs:
            log.close()
        seed_everything(42)
        for curve in curves:
            config = next(r['config'] for r in runs if r['output'] == curve['output'])
            curve['checkpoint_evaluation'] = {name: checkpoint_metrics(Path(curve['output'])/f'{name}_model.pt', config)
                                                for name in ('best', 'final')}
        for seed, _, run_root in jobs:
            directory = Path(read(run_root/'result.json')['output'])
            metadata, metrics = read(directory/'experiment.json'), read(directory/'metrics.json')
            history = [json.loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
            if metadata['status'] != 'completed' or len(history) != 300 or metrics['test_evaluated']:
                raise RuntimeError('Incomplete centralized experiment')
            reference = next(r for r in runs if r['config']['seed'] == seed)
            for key in ('manifest_sha256', 'versions', 'python', 'platform', 'reproducibility', 'initial_model_sha256'):
                if metadata[key] != reference['metadata'][key]:
                    raise RuntimeError(f'Centralized/reference metadata mismatch: {key}')
            evaluations = {name: checkpoint_metrics(directory/f'{name}_model.pt', metrics['config']) for name in ('best', 'final')}
            selected = min(history, key=lambda r: r['validation_loss'])
            for name, row in (('best', selected), ('final', history[-1])):
                if evaluations[name]['model_sha256'] != row['model_sha256'] or evaluations[name]['step'] != row['step']:
                    raise RuntimeError('Centralized checkpoint hash/selection mismatch')
            curves.append(dict(seed=seed, method='centralized', output=str(directory), best=metrics['best_checkpoint'],
                history=history, checkpoint_evaluation=evaluations))
        if source_hash(ROOT) != source:
            raise RuntimeError('Working training source changed during diagnosis')
        config = runs[0]['config']
        manifest = read_manifest(config['manifest'], config['num-classes'], config['num-clients'])
        counts = Counter(r['client_id'] for r in manifest['examples'] if r['split'] == 'train')
        rounds, batch = config['num-server-rounds'], config['batch-size']
        record.update(status='completed', curves_file='curves.json',
            budget=dict(training_examples=sum(counts.values()), validation_examples=sum(r['split']=='validation' for r in manifest['examples']),
                presentations_per_method=sum(counts.values())*rounds*config['local-epochs'],
                centralized_sgd_steps=math.ceil(sum(counts.values())/batch)*rounds*config['local-epochs'],
                federated_sgd_steps_all_clients=sum(math.ceil(n/batch) for n in counts.values())*rounds*config['local-epochs'],
                federated_aggregation_steps=rounds),
            methods={method: dict(selected_validation_accuracy_mean=statistics.mean(c['best']['validation_accuracy'] for c in curves if c['method']==method),
                selected_validation_accuracy_sample_sd=statistics.stdev(c['best']['validation_accuracy'] for c in curves if c['method']==method))
                for method in ('none', 'qsgd', 'centralized')},
            notes=['Same model/training code, fixed split and matched initialization per seed.',
                'Centralized shuffles the pooled training set; federated shuffles each client separately each round.',
                'Federated optimization loss is measured on changing local models; centralized loss on one evolving pooled model.',
                'Source-wide hashes differ from the earlier study due to LLZ integration; task.py is byte-identical and QSGD replay is exact.',
                'No communication or wall-time fairness claim; centralized runs use three concurrent independent one-thread processes.'])
        write_json(output/'curves.json', dict(status='completed', curves=curves))
        write_json(output/'diagnosis.json', record)
        print(f'Completed diagnosis: {output / "diagnosis.json"}', flush=True)
    except BaseException as error:
        for _, child, _ in jobs:
            if child.poll() is None:
                child.terminate()
        for _, child, _ in jobs:
            child.wait()
        for log in logs:
            log.close()
        record.update(status='failed', error=str(error))
        write_json(output/'diagnosis.json', record)
        raise


if __name__ == '__main__':
    main()
