"""Resumable seed-42 CIFAR-10 LLZ batch and an update-capturing task adapter.

The adapter delegates model construction, data loading, training and evaluation
to cifar10_task. Only the lossless run writes training-update snapshots. The
controller supervises the existing Flower launcher and accepts completion only
after the saved experiment, history, communication and checkpoints pass audit.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import tomllib
from importlib.metadata import version

import numpy as np
import torch

from compression.adaptive_llz import admissible, select
from flower_face import cifar10_task
from flower_face.reproducibility import source_hash
from flower_face.run import launch
from flower_face.study import collect, snapshot, write_json


TASK_MODULE = 'flower_face.cifar10_batch'
MODULE_FILE = Path('flower_face/cifar10_batch.py')
ROUND_TIMEOUT = cifar10_task.ROUND_TIMEOUT
Net = cifar10_task.Net
read_manifest = cifar10_task.read_manifest
validate_config = cifar10_task.validate_config
test = cifar10_task.test


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path, *, allow_partial=False):
    text = Path(path).read_text()
    lines = text.splitlines()
    if allow_partial and text and not text.endswith('\n'):
        lines = lines[:-1]
    return [json.loads(line) for line in lines]


def capture_rounds(rounds):
    return sorted({1, (rounds + 1) // 2, rounds})


def load_data(config, client_id=None, split='train', seed=None, *, apply_augmentation=True):
    loader = cifar10_task.load_data(config, client_id, split, seed,
                                  apply_augmentation=apply_augmentation)
    if (split == 'train' and seed is not None and client_id is not None
            and config['compression'] == 'qsgd-llz' and config['llz-p'] == 0):
        round_number, remainder = divmod(seed - config['seed'] - client_id, 1000)
        if remainder != 0 or not 1 <= round_number <= config['num-server-rounds']:
            raise ValueError('Capture cannot identify the configured client training round')
        if round_number in capture_rounds(config['num-server-rounds']):
            loader.capture_info = (Path(config['output-dir']) / 'calibration-updates',
                                   round_number, client_id)
    return loader


def train(model, loader, epochs, learning_rate, *, weight_decay=0.0):
    capture = getattr(loader, 'capture_info', None)
    initial = None
    if capture:
        # Keep the received model intact so we can save the exact local delta.
        initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    loss = cifar10_task.train(model, loader, epochs, learning_rate, weight_decay=weight_decay)
    if capture:
        folder, round_number, client = capture
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f'round-{round_number:02d}-client-{client}.npz'
        temporary = target.with_suffix('.npz.tmp')
        updates = {name: (value.detach().cpu() - initial[name]).numpy()
                   for name, value in model.state_dict().items()}
        with temporary.open('wb') as stream:
            np.savez(stream, **updates)
        os.replace(temporary, target)
        write_json(target.with_suffix('.json'), dict(round=round_number, client=client,
                   sha256=sha256(target), bytes=target.stat().st_size,
                   train_loss=float(loss), num_examples=len(loader.dataset)))
    return loss


def original_source_hash(root):
    """Identify pre-existing training code separately from this new adapter."""
    entries = {}
    files = [root / 'pyproject.toml']
    for package in ('flower_face', 'compression', 'federated_compression'):
        files.extend((root / package).rglob('*.py'))
    for path in sorted(files):
        if path.relative_to(root) == MODULE_FILE:
            continue
        content = (json.dumps(tomllib.loads(path.read_text()), sort_keys=True).encode()
                   if path.name == 'pyproject.toml' else path.read_bytes())
        entries[path.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def collect_reference(folder):
    metadata = load_json(folder / 'experiment.json')
    result = collect(folder.parent, metadata['config'])
    if Path(result['output']).resolve() != folder.resolve():
        raise ValueError('Reference path resolved to a different experiment')
    return result


def check_match(reference, candidate):
    for key in ('manifest_sha256', 'initial_model_sha256', 'versions', 'python', 'platform',
                'reproducibility', 'fixed_state_policy', 'selection_rule', 'communication_measurement'):
        if reference['metadata'][key] != candidate['metadata'][key]:
            raise ValueError(f'Matched run differs in {key}')
    # The verified manifest hash identifies the split across different filesystem roots.
    excluded = {'task-module', 'manifest', 'output-dir', 'compression', 'llz-p', 'llz-budget'}
    if ({key: value for key, value in reference['config'].items() if key not in excluded}
            != {key: value for key, value in candidate['config'].items() if key not in excluded}):
        raise ValueError('Matched training settings differ')


def exact_replay(reference, candidate):
    check_match(reference, candidate)
    def learning(result):
        return [dict(round=row['round'], model_sha256=row['model_sha256'], validation=row['validation'],
                     train={key: value for key, value in row['train'].items()
                            if not key.startswith(('qsgd_', 'llz_', 'adaptive_'))})
                for row in result['replay']]
    return learning(reference) == learning(candidate) and reference['best'] == candidate['best']


def stop_group(process):
    """Terminate only the launcher's private process group after saved completion."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def adopt_detached_children():
    """Keep orphaned Flower services owned by this Linux batch controller.

    Flower starts SuperLink/SuperExec in separate sessions. A process-group kill
    alone cannot reach them. Linux PR_SET_CHILD_SUBREAPER reparents those orphans
    to this controller, so cleanup can use ownership rather than process names.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot supervise detached Flower services')


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return int(fields[19]), int(fields[1]), fields[0]  # start time, parent PID, state


def owned_children(protected):
    """Enumerate this controller's new descendants, preserving earlier trees."""
    pending, found = [os.getpid()], {}
    while pending:
        parent = pending.pop()
        # Services may create subprocesses from threads other than their leader.
        try:
            children_files = list(Path(f'/proc/{parent}/task').glob('*/children'))
        except FileNotFoundError:
            # The process can exit between discovering it and enumerating its threads.
            continue
        for children_file in children_files:
            try:
                children = children_file.read_text().split()
            except FileNotFoundError:
                continue
            for text in children:
                pid = int(text)
                identity = process_identity(pid)
                if identity is None or protected.get(pid) == identity[0] or pid in found:
                    continue
                found[pid] = identity[0]
                pending.append(pid)
    return found


def cleanup_detached_children(protected):
    signalled = set()
    for sig, grace in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
        targets = owned_children(protected)
        for pid, started in reversed(list(targets.items())):
            identity = process_identity(pid)
            if identity is not None and identity[0] == started:
                try:
                    os.kill(pid, sig)
                    signalled.add(pid)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + grace
        while True:
            remaining = owned_children(protected)
            for pid, started in remaining.items():
                identity = process_identity(pid)
                if identity is not None and identity == (started, os.getpid(), 'Z'):
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            if not owned_children(protected):
                return len(signalled)
            if time.monotonic() >= deadline:
                break
            time.sleep(.05)
    raise RuntimeError('Some owned Flower services survived cleanup; inspect the job before restarting')


def failure_detail(run_root):
    chunks = []
    for name in ('terminal.log', 'worker.log'):
        file = run_root / name
        if file.is_file() and file.stat().st_size:
            chunks.append(f'{file}\n' + '\n'.join(file.read_text(errors='replace').splitlines()[-35:]))
    return '\n'.join(chunks)


def supervise(root, stage, run_root, config, *, completion_grace=30, poll_seconds=10):
    adopt_detached_children()
    protected = owned_children({})
    config_path = run_root / 'requested-config.json'
    write_json(config_path, config)
    command = [sys.executable, '-u', '-m', TASK_MODULE, '--worker', str(config_path), '--app', str(stage)]
    started = time.monotonic()
    completion_time = None
    last_round = -1
    cleaned = False
    result = None
    error = None
    exit_seen = None
    with (run_root / 'worker.log').open('a') as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while True:
                folders = [path for path in run_root.iterdir() if path.is_dir() and (path / 'experiment.json').is_file()]
                if len(folders) > 1:
                    raise RuntimeError('More than one experiment in an isolated attempt directory')
                metadata = load_json(folders[0] / 'experiment.json') if folders else {}
                history = read_rows(folders[0] / 'history.jsonl', allow_partial=True) if folders and (folders[0] / 'history.jsonl').exists() else []
                if history and history[-1]['round'] != last_round:
                    last_round = history[-1]['round']
                    print(f'  round {last_round}/{config["num-server-rounds"]}, validation {history[-1]["validation"]["eval_acc"]:.2%}', flush=True)
                return_code = process.poll()
                # The launcher may finish between the metadata read and poll().
                if return_code is not None and folders:
                    metadata = load_json(folders[0] / 'experiment.json')
                if metadata.get('status') == 'completed':
                    if completion_time is None:
                        completion_time = time.monotonic()
                        result = collect(run_root, config)
                    if return_code is not None:
                        break
                    if time.monotonic() - completion_time >= completion_grace:
                        cleaned = True
                        break
                elif return_code is not None:
                    if exit_seen is None:
                        exit_seen = time.monotonic()
                    if return_code != 0 or time.monotonic() - exit_seen >= 2:
                        raise RuntimeError(f'Launcher exited {return_code} before completion.\n{failure_detail(run_root)}')
                elif not folders and time.monotonic() - started > 1800:
                    raise RuntimeError(f'No experiment started after 30 minutes: {run_root / "terminal.log"}')
                time.sleep(poll_seconds)
        except BaseException as failure:
            error = str(failure)
            raise
        finally:
            if process.poll() is None:
                stop_group(process)
            detached = cleanup_detached_children(protected)
            write_json(run_root / 'launcher-status.json', dict(completed_artifacts_verified=result is not None,
                       cleanup_after_completion=result is not None and (cleaned or detached > 0),
                       detached_processes_signalled=detached, error=error,
                       launcher_exit_code=process.returncode, launch_seconds=time.monotonic() - started))
    return result


def execute(root, output, stage, protocol, label, compression, budget=0.0):
    parent = output / label
    parent.mkdir(exist_ok=True)
    settings = dict(protocol['config'])
    settings.update({'task-module': TASK_MODULE, 'compression': compression, 'llz-p': 0, 'llz-budget': float(budget)})
    attempts = sorted(parent.glob('attempt-*'))
    for attempt in attempts:
        expected = dict(settings, **{'output-dir': str(attempt)})
        completed = [path for path in attempt.glob('*/experiment.json')
                     if load_json(path).get('status') == 'completed']
        if completed:
            result = collect(attempt, expected)
            if result['metadata']['source_sha256'] != protocol['source_sha256']:
                raise RuntimeError('Completed attempt source differs from frozen protocol')
            print(f'Reusing completed {label}', flush=True)
            return result
    attempt = parent / f'attempt-{len(attempts) + 1:02d}'
    attempt.mkdir(exist_ok=False)
    settings['output-dir'] = str(attempt)
    print(f'Starting {label}, budget={budget:g}', flush=True)
    result = supervise(root, stage, attempt, settings)
    if result['metadata']['source_sha256'] != protocol['source_sha256']:
        raise RuntimeError('Training source differs from frozen protocol')
    print(f'Completed {label}: validation={result["best"]["validation_accuracy"]:.2%}, training uploads={result["train_upload_bytes"] / 1e6:.1f} MB', flush=True)
    return result


def calibrate(output, protocol, lossless):
    destination = output / 'calibration.json'
    snapshots = Path(lossless['output']).parent / 'calibration-updates'
    config = protocol['config']
    sampled_rounds = capture_rounds(config['num-server-rounds'])
    expected = {
        (round_number, client)
        for round_number in sampled_rounds
        for client in range(config['num-clients'])
    }
    saved = {}
    if destination.exists():
        saved = load_json(destination)
        if saved['source_sha256'] != protocol['source_sha256']:
            raise RuntimeError('Calibration source differs from protocol')
        records = saved['updates']
    else:
        records = []
    saved_updates = {(record['round'], record['client']): record for record in records}
    if len(saved_updates) != len(records) or not saved_updates.keys() <= expected:
        raise RuntimeError('Unexpected or duplicate calibration records')

    for round_number, client in sorted(expected):
        snapshot_path = snapshots / f'round-{round_number:02d}-client-{client}.npz'
        info = load_json(snapshot_path.with_suffix('.json'))
        if ((info['round'], info['client']) != (round_number, client)
                or sha256(snapshot_path) != info['sha256']
                or snapshot_path.stat().st_size != info['bytes']):
            raise RuntimeError('Captured update checksum or identity differs')
        existing = saved_updates.get((round_number, client))
        if existing is not None:
            if existing['snapshot_sha256'] != info['sha256']:
                raise RuntimeError('Reused snapshot has changed')
            continue
        with np.load(snapshot_path, allow_pickle=False) as archive:
            updates = {name: archive[name] for name in archive.files}
        # Reproduce the QSGD draw the client used for this round and partition.
        seed_sequence = np.random.SeedSequence([config['seed'], round_number, client, 0x51534744])
        selection = select(
            updates, levels=config['qsgd-levels'], rng=np.random.default_rng(seed_sequence),
            budget=0.0, window_size=config['llz-window'],
        )
        update_norm = selection.candidates[0].stats['qsgd_input_squared_norm']
        candidates = []
        for candidate in selection.candidates:
            added_error = candidate.stats['llz_reconstruction_squared_error']
            candidates.append(dict(p=candidate.p, payload_bytes=candidate.payload_bytes,
                                   **candidate.stats,
                                   extra_error_ratio=added_error / update_norm if update_norm else 0.0))
        if (candidates[0]['extra_error_ratio'] != 0
                or any(candidate['llz_max_code_error'] > candidate['p'] for candidate in candidates)):
            raise RuntimeError('Calibration codec invariant failed')
        records.append(dict(round=round_number, client=client, snapshot_sha256=info['sha256'], candidates=candidates))
        write_json(destination, dict(source_sha256=protocol['source_sha256'], updates=records))
        print(f'Calibrated update round {round_number}, client {client}: p1 ratio={candidates[1]["extra_error_ratio"]:.6g}', flush=True)

    # Match the saved snapshots to the lossless run's aggregated diagnostics.
    for round_number in sampled_rounds:
        round_metrics = lossless['replay'][round_number - 1]['train']
        for field in ('qsgd_input_squared_norm', 'qsgd_quantization_squared_error'):
            total = sum(record['candidates'][0][field] for record in records if record['round'] == round_number)
            if not math.isclose(total, round_metrics[field], rel_tol=1e-10, abs_tol=1e-12):
                raise RuntimeError(f'Captured update {field} differs from actual lossless run')

    p1_ratios = np.array([record['candidates'][1]['extra_error_ratio'] for record in records])
    budgets = [
        dict(coverage_percent=coverage, budget=float(np.quantile(p1_ratios, coverage / 100, method='linear')))
        for coverage in protocol['budget_coverage_percent']
    ]
    if any(not math.isfinite(item['budget']) or item['budget'] < 0 for item in budgets):
        raise RuntimeError('Calibration returned an invalid budget')
    for item in budgets:
        item['observed_p1_feasible_fraction'] = float(np.mean(p1_ratios <= item['budget']))
    saved = dict(source_sha256=protocol['source_sha256'], updates=records, budgets=budgets,
                 completed_at_utc=saved.get('completed_at_utc', datetime.now(timezone.utc).isoformat()),
                 scope='Twelve clustered training updates on a lossless seed-42 trajectory; codec packet costs, including headers. Full training uses serialized reply costs.')
    write_json(destination, saved)
    return saved


def audit_adaptive(run):
    folder = Path(run['output'])
    rows = read_rows(folder / 'adaptive_selection.jsonl')
    uploads = {(row['round'], row['client_partition_id']): row
               for row in read_rows(folder / 'communication_messages.jsonl')
               if row['phase'] == 'train' and row['direction'] == 'uplink'}
    expected = {(number, client) for number in range(1, run['config']['num-server-rounds'] + 1)
                for client in range(run['config']['num-clients'])}
    if len(rows) != len(expected) or {(row['round'], row['client']) for row in rows} != expected:
        raise RuntimeError('Adaptive diagnostics are incomplete')
    for row in rows:
        norm = row['qsgd_input_squared_norm']
        errors, costs = row['adaptive_candidate_squared_error'], row['adaptive_candidate_message_bytes']
        p = row['adaptive_selected_p']
        feasible = [value for value in (0, 1, 2) if admissible(errors[value], norm, run['config']['llz-budget'])]
        actual = uploads[row['round'], row['client']]
        if (errors[0] != 0 or not feasible or p != min(feasible, key=lambda value: (costs[value], value))
                or row['adaptive_budget'] != run['config']['llz-budget']
                or not admissible(row['llz_reconstruction_squared_error'], norm, row['adaptive_budget'])
                or row['observed_message_bytes'] != actual['serialized_object_bytes']
                or row['adaptive_selected_message_bytes'] != row['observed_message_bytes']
                or row['adaptive_candidate_payload_bytes'][p] != actual['codec_payload_bytes']):
            raise RuntimeError('Adaptive decision or byte audit failed')
    return dict(selected_p_counts={str(p): sum(row['adaptive_selected_p'] == p for row in rows) for p in (0, 1, 2)},
                max_extra_error_ratio=max(row['llz_reconstruction_squared_error'] / row['qsgd_input_squared_norm']
                                         if row['qsgd_input_squared_norm'] else 0.0 for row in rows),
                selection_seconds_sum=sum(row['adaptive_selection_seconds'] for row in rows),
                decode_seconds_sum=sum(row['decode_seconds'] for row in rows))


def report(output, reference, lossless, adaptive, gate):
    methods = {'qsgd': reference, 'llz-p0': lossless, **adaptive}
    summaries = {}
    for label, run in methods.items():
        check_match(reference, run)
        drop = 100 * (reference['best']['validation_accuracy'] - run['best']['validation_accuracy'])
        row = dict(output=run['output'], validation_accuracy_percent=100 * run['best']['validation_accuracy'],
                   selected_round=run['best']['step'], accuracy_drop_vs_qsgd_pp=drop,
                   within_accuracy_margin=drop <= gate + 1e-12,
                   training_upload_bytes=run['train_upload_bytes'], total_serialized_bytes=run['total_bytes'],
                   upload_saving_vs_qsgd_percent=100 * (1 - run['train_upload_bytes'] / reference['train_upload_bytes']),
                   upload_saving_vs_p0_percent=100 * (1 - run['train_upload_bytes'] / lossless['train_upload_bytes']),
                   training_elapsed_seconds=run['metadata']['elapsed_seconds'])
        if label.startswith('adaptive-'):
            row.update(budget=run['config']['llz-budget'], **run['selection_audit'])
        summaries[label] = row
    eligible = [label for label in adaptive if summaries[label]['within_accuracy_margin']]
    chosen = min(eligible, key=lambda label: (summaries[label]['training_upload_bytes'], summaries[label]['budget'])) if eligible else None
    summary = dict(status='completed', seed=42, accuracy_margin_pp=gate, exact_p0_replay=True,
                   selected_development_policy=chosen, methods=summaries,
                   scope='Seed-42 development only. Selected checkpoints use minimum validation loss. Held-out test unused. Further seeds require a frozen policy. Logical serialized bytes exclude transport.')
    write_json(output / 'summary.json', summary)
    import csv
    with (output / 'comparison.csv').open('w', newline='') as stream:
        columns = ['method', 'budget', 'validation_accuracy_percent', 'accuracy_drop_vs_qsgd_pp', 'within_accuracy_margin',
                   'training_upload_bytes', 'upload_saving_vs_p0_percent', 'training_elapsed_seconds']
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(dict(method=label, **row) for label, row in summaries.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, run in methods.items():
        rows = run['replay']
        history = read_rows(Path(run['output']) / 'history.jsonl')
        axes[0].plot([row['round'] for row in rows],
                     [row['communication']['train']['uplink']['codec_payload_bytes'] /
                      (4 * row['train']['qsgd_coordinate_count']) * 100 for row in history], label=label)
        axes[1].plot([row['round'] for row in rows], [row['train']['train_loss'] for row in rows], label=label)
        axes[2].plot([row['round'] for row in rows], [100 * row['validation']['eval_acc'] for row in rows], label=label)
    for ax, title, ylabel in zip(axes, ['(a) Codec size', '(b) Training loss', '(c) Validation accuracy'],
                                ['Codec payload / raw FP32 (%)', 'Mean client loss', 'Accuracy (%)']):
        ax.set(title=title, xlabel='Communication round', ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0].set_yscale('log')
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    fig.suptitle('CIFAR-10 adaptive LLZ development, seed 42')
    fig.tight_layout(rect=(0, .14, 1, .95))
    fig.savefig(output / 'learning-and-compression.png', dpi=200)
    plt.close(fig)
    print('Batch completed. Summary, comparison table and graph saved to', output, flush=True)
    return summary


def environment():
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.machine())
    return dict(platform=platform.platform(), python=platform.python_version(), cpu_model=cpu,
                versions={name: version(name) for name in ('flwr', 'torch', 'torchvision', 'numpy', 'ray')})


def package_results(output, methods):
    import zipfile
    with zipfile.ZipFile(output / 'results-for-review.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('protocol.json', 'frozen-budgets.json', 'calibration.json', 'summary.json',
                     'comparison.csv', 'learning-and-compression.png', 'batch-status.json'):
            archive.write(output / name, name)
        for label, run in methods.items():
            folder = Path(run['output'])
            for name in ('experiment.json', 'best_checkpoint.json', 'history.jsonl', 'metrics.json',
                         'communication.json', 'communication_messages.jsonl', 'adaptive_selection.jsonl'):
                if (folder / name).exists():
                    archive.write(folder / name, f'runs/{label}/{name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--manifest', type=Path, help='Local copy of the reference CIFAR-10 manifest')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/cifar10-adaptive-batch-seed42-v2'))
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--app', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker:
        config = load_json(args.worker)
        raise SystemExit(launch(config, root=root, app_dir=args.app,
                               log_path=Path(config['output-dir']) / 'terminal.log'))
    if args.reference is None or not sys.platform.startswith('linux'):
        parser.error('Use Linux/WSL and --reference pointing to the completed HPC QSGD experiment')
    torch.set_num_threads(1)
    reference = collect_reference(args.reference.resolve())
    cfg = dict(reference['config'])
    if (cfg['task-module'], cfg['compression'], cfg['seed'], cfg['num-server-rounds'],
            cfg['num-clients'], cfg['learning-rate'], cfg['qsgd-levels'], cfg['llz-window']) != (
                'flower_face.cifar10_task', 'qsgd', 42, 15, 4, .03, 127, 128):
        raise ValueError('Expected the frozen 15-round CIFAR-10 HPC QSGD reference')
    if original_source_hash(root) != reference['metadata']['source_sha256']:
        raise ValueError('Existing training source differs from the completed reference; inspect the source before running')
    manifest = (args.manifest or Path(cfg['manifest'])).resolve()
    if not manifest.is_file():
        parser.error(f'Manifest unavailable at {manifest}. Use --manifest with the prepared local copy.')
    if sha256(manifest) != reference['metadata']['manifest_sha256']:
        raise ValueError('The manifest checksum differs from the reference CIFAR-10 split')
    cfg['manifest'] = str(manifest)
    output = args.output_dir.resolve()
    if not output.is_relative_to(root / 'outputs'):
        raise ValueError('Batch results must stay inside the project outputs directory')
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / '.batch.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This batch is already running')
        stage = output / 'app'
        protocol_path = output / 'protocol.json'
        if protocol_path.exists():
            protocol = load_json(protocol_path)
            if (source_hash(root) != protocol['source_sha256'] or source_hash(stage) != protocol['source_sha256']
                    or str(args.reference.resolve()) != protocol['reference']
                    or cfg != protocol['config'] or reference['metadata']['manifest_sha256'] != protocol['manifest_sha256']
                    or environment() != protocol['environment']):
                raise RuntimeError('Frozen source, reference, manifest or batch environment changed')
        else:
            snapshot(root, stage)
            protocol = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), config=cfg,
                            source_sha256=source_hash(stage), manifest_sha256=reference['metadata']['manifest_sha256'],
                            original_training_source_sha256=reference['metadata']['source_sha256'],
                            reference=str(args.reference.resolve()), accuracy_margin_pp=1.0,
                            capture_rounds=capture_rounds(15), capture_clients=list(range(4)),
                            budget_coverage_percent=[25, 50, 75],
                            budget_rule='Linear quantiles Q25/Q50/Q75 of p=1 additional-error ratios on 12 captured training updates; full precision, no hand rounding. Deduplicate equal candidates.',
                            selection_rule='Among candidates within 1.0 validation percentage point of matched QSGD, choose minimum training upload bytes, then smaller budget.',
                            reference_fallback='If historical-reference exact replay differs, run one QSGD reference on this allocated node and require exact replay with lossless LLZ.',
                            environment=environment())
            write_json(protocol_path, protocol)
        print('Frozen batch:', output, flush=True)
        try:
            lossless = execute(root, output, stage, protocol, 'llz-p0', 'qsgd-llz')
            try:
                historical_replay = exact_replay(reference, lossless)
            except ValueError:
                historical_replay = False
            if not historical_replay:
                print('Historical HPC trajectory differs. Establishing a QSGD reference on this node.', flush=True)
                reference = execute(root, output, stage, protocol, 'qsgd-node-reference', 'qsgd')
                if not exact_replay(reference, lossless):
                    raise RuntimeError('Same-node lossless replay failed; saved results retained for diagnosis')
            print('All-round lossless replay passed.', flush=True)
            calibration = calibrate(output, protocol, lossless)
            frozen_budgets = dict(budgets=calibration['budgets'], accuracy_margin_pp=1.0,
                                  calibration_sha256=sha256(output / 'calibration.json'), rule=protocol['budget_rule'])
            frozen_path = output / 'frozen-budgets.json'
            if frozen_path.exists():
                if load_json(frozen_path) != frozen_budgets:
                    raise RuntimeError('Frozen budget evidence changed')
            else:
                write_json(frozen_path, frozen_budgets)
            adaptive = {}
            by_budget = {}
            for point in calibration['budgets']:
                label = f'adaptive-q{point["coverage_percent"]}'
                budget = point['budget']
                if budget in by_budget:
                    print(f'{label} shares an already tested budget; reusing its result.', flush=True)
                    continue
                run = execute(root, output, stage, protocol, label, 'qsgd-llz-adaptive', budget)
                check_match(reference, run)
                run['selection_audit'] = audit_adaptive(run)
                adaptive[label] = run
                by_budget[budget] = label
            report(output, reference, lossless, adaptive, protocol['accuracy_margin_pp'])
            write_json(output / 'batch-status.json', dict(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat()))
            package_results(output, {'qsgd': reference, 'llz-p0': lossless, **adaptive})
        except BaseException as error:
            write_json(output / 'batch-status.json', dict(status='failed', error=str(error)))
            raise


if __name__ == '__main__':
    main()
