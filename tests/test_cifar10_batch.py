import copy
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from flower_face import cifar10_batch as batch
from flower_face.task import seed_everything
from flower_face.task_api import resolve


def test_capture_task_keeps_the_cifar_client_timeout():
    task = resolve({'task-module': batch.TASK_MODULE})
    assert task.ROUND_TIMEOUT == batch.cifar10_task.ROUND_TIMEOUT == 3600


def test_capture_preserves_training_and_random_streams(tmp_path):
    seed_everything(89)
    data = TensorDataset(torch.randn(12, 8), torch.randint(0, 3, (12,)))
    initial = torch.nn.Linear(8, 3).state_dict()

    def execute(capture):
        model = torch.nn.Linear(8, 3)
        model.load_state_dict(initial)
        loader = DataLoader(data, batch_size=4, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        if capture:
            loader.capture_info = (tmp_path, 8, 2)
        seed_everything(100)
        loss = batch.train(model, loader, 1, .03)
        streams = (torch.rand(8), np.random.random(8), random.random())
        return loss, copy.deepcopy(model.state_dict()), streams

    ordinary, captured = execute(False), execute(True)
    assert ordinary[0] == captured[0]
    assert all(torch.equal(ordinary[1][key], captured[1][key]) for key in ordinary[1])
    assert torch.equal(ordinary[2][0], captured[2][0])
    np.testing.assert_array_equal(ordinary[2][1], captured[2][1])
    assert ordinary[2][2] == captured[2][2]
    path = tmp_path / 'round-08-client-2.npz'
    with np.load(path, allow_pickle=False) as saved:
        for name in initial:
            np.testing.assert_array_equal(saved[name], (captured[1][name] - initial[name]).numpy())
    assert json.loads(path.with_suffix('.json').read_text())['sha256'] == batch.sha256(path)


@pytest.mark.parametrize('round_number,compression,expected', [
    (1, 'qsgd-llz', True), (8, 'qsgd-llz', True), (15, 'qsgd-llz', True),
    (7, 'qsgd-llz', False), (8, 'qsgd-llz-adaptive', False), (8, 'qsgd', False),
])
def test_capture_selects_only_lossless_early_middle_late_updates(monkeypatch, tmp_path, round_number, compression, expected):
    monkeypatch.setattr(batch.cifar10_task, 'load_data', lambda *args, **kwargs: SimpleNamespace())
    config = {'seed': 42, 'num-server-rounds': 15, 'compression': compression, 'llz-p': 0,
              'output-dir': str(tmp_path)}
    loader = batch.load_data(config, 3, 'train', 42 + 1000 * round_number + 3)
    assert hasattr(loader, 'capture_info') == expected


def test_partial_history_is_tolerated_only_while_polling(tmp_path):
    file = tmp_path / 'history.jsonl'
    file.write_text('{"round": 1}\n{"round":')
    assert batch.read_rows(file, allow_partial=True) == [{'round': 1}]
    with pytest.raises(json.JSONDecodeError):
        batch.read_rows(file)


def test_child_scan_continues_when_a_process_exits_during_glob(monkeypatch):
    root = batch.os.getpid()
    exiting, surviving, grandchild = root + 100, root + 101, root + 102

    class ProcTask:
        def __init__(self, path):
            self.pid = int(path.split('/')[2])

        def glob(self, pattern):
            assert pattern == '*/children'
            if self.pid == exiting:
                raise FileNotFoundError(self.pid)
            if self.pid == root:
                yield SimpleNamespace(read_text=lambda: f'{surviving} {exiting}')
            if self.pid == surviving:
                yield SimpleNamespace(read_text=lambda: str(grandchild))

    monkeypatch.setattr(batch, 'Path', ProcTask)
    monkeypatch.setattr(batch, 'process_identity', lambda pid: (pid * 10, root, 'S'))
    assert batch.owned_children({}) == {pid: pid * 10 for pid in (exiting, surviving, grandchild)}


@pytest.mark.skipif(sys.platform == 'win32', reason='The HPC supervisor uses POSIX process groups')
def test_completed_training_stops_a_lingering_launcher(monkeypatch, tmp_path):
    folder = tmp_path / 'experiment'
    folder.mkdir()
    (folder / 'experiment.json').write_text('{"status":"completed"}')
    create = subprocess.Popen
    children = []

    def sleeping_launcher(command, **kwargs):
        child = create([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(batch.subprocess, 'Popen', sleeping_launcher)
    monkeypatch.setattr(batch, 'collect', lambda root, expected: {'audited': True})
    result = batch.supervise(tmp_path, tmp_path, tmp_path, {'num-server-rounds': 1},
                             completion_grace=.02, poll_seconds=.01)
    assert result == {'audited': True}
    assert children[0].poll() is not None
    assert json.loads((tmp_path / 'launcher-status.json').read_text())['cleanup_after_completion']


@pytest.mark.skipif(sys.platform == 'win32', reason='The HPC supervisor uses POSIX process groups')
def test_failed_launcher_is_not_reported_as_completed(monkeypatch, tmp_path):
    create = subprocess.Popen
    monkeypatch.setattr(batch.subprocess, 'Popen',
                        lambda command, **kwargs: create([sys.executable, '-c', 'raise SystemExit(2)'], **kwargs))
    with pytest.raises(RuntimeError, match='before completion'):
        batch.supervise(tmp_path, tmp_path, tmp_path, {'num-server-rounds': 1}, poll_seconds=.01)
    status = json.loads((tmp_path / 'launcher-status.json').read_text())
    assert not status['completed_artifacts_verified']
    assert status['launcher_exit_code'] == 2


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux detached service supervision')
@pytest.mark.parametrize('ignore_term', [False, True])
def test_failed_launcher_cleans_detached_services_and_preserves_existing_processes(monkeypatch, tmp_path, ignore_term):
    create = subprocess.Popen
    pid_file, ready_file = tmp_path / 'detached.pid', tmp_path / 'ready'
    service = f'''import pathlib, signal, time
if {ignore_term!r}:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path({str(ready_file)!r}).write_text('ready')
time.sleep(60)
'''
    launcher = f'''import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', {service!r}], start_new_session=True)
pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))
deadline = time.monotonic() + 5
while not pathlib.Path({str(ready_file)!r}).exists():
    if time.monotonic() > deadline:
        raise SystemExit(2)
    time.sleep(.01)
'''
    foreign = create([sys.executable, '-c', 'import time; time.sleep(60)'])
    monkeypatch.setattr(batch.subprocess, 'Popen', lambda command, **kwargs: create([sys.executable, '-c', launcher], **kwargs))
    (tmp_path / 'terminal.log').write_text('ERROR: client reply deadline exceeded\n')
    try:
        with pytest.raises(RuntimeError, match='client reply deadline exceeded'):
            batch.supervise(tmp_path, tmp_path, tmp_path, {'num-server-rounds': 1}, poll_seconds=.01)
        assert batch.process_identity(int(pid_file.read_text())) is None
        assert foreign.poll() is None
        status = json.loads((tmp_path / 'launcher-status.json').read_text())
        assert status['detached_processes_signalled'] >= 1
        assert not status['completed_artifacts_verified']
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux supervisor')
def test_launcher_exit_rechecks_completion_metadata(monkeypatch, tmp_path):
    folder = tmp_path / 'experiment'
    folder.mkdir()
    metadata = folder / 'experiment.json'
    metadata.write_text('{"status":"incomplete"}')

    class FinishedLauncher:
        returncode = 0

        def poll(self):
            metadata.write_text('{"status":"completed"}')
            return 0

    monkeypatch.setattr(batch.subprocess, 'Popen', lambda *args, **kwargs: FinishedLauncher())
    monkeypatch.setattr(batch, 'collect', lambda root, expected: {'audited': True})
    assert batch.supervise(tmp_path, tmp_path, tmp_path, {'num-server-rounds': 1}, poll_seconds=.01) == {'audited': True}


def test_calibration_uses_all_clients_and_rounds_and_reuses_the_freeze(tmp_path):
    output = tmp_path / 'batch'
    output.mkdir()
    attempt = output / 'lossless'
    snapshots = attempt / 'calibration-updates'
    snapshots.mkdir(parents=True)
    config = {'seed': 42, 'num-server-rounds': 15, 'num-clients': 4, 'qsgd-levels': 127, 'llz-window': 128}
    protocol = {'config': config, 'source_sha256': 'frozen', 'budget_coverage_percent': [25, 50, 75]}
    lossless = {'output': str(attempt / 'experiment'), 'replay': [{'train': {}} for _ in range(15)]}
    fields = ('qsgd_input_squared_norm', 'qsgd_quantization_squared_error')
    expected_errors = []
    for number in (1, 8, 15):
        total = dict.fromkeys(fields, 0.0)
        for client in range(4):
            updates = {'weight': np.random.default_rng(number * 4 + client).normal(size=64).astype('float32')}
            path = snapshots / f'round-{number:02d}-client-{client}.npz'
            np.savez(path, **updates)
            path.with_suffix('.json').write_text(json.dumps(dict(round=number, client=client,
                sha256=batch.sha256(path), bytes=path.stat().st_size)))
            result = batch.select(updates, levels=127, rng=np.random.default_rng(
                np.random.SeedSequence([42, number, client, 0x51534744])), budget=0.0, window_size=128)
            norm = result.candidates[0].stats['qsgd_input_squared_norm']
            expected_errors.append(result.candidates[1].stats['llz_reconstruction_squared_error'] / norm)
            for field in fields:
                total[field] += result.candidates[0].stats[field]
        lossless['replay'][number - 1]['train'] = total
    calibrated = batch.calibrate(output, protocol, lossless)
    assert len(calibrated['updates']) == 12
    assert [row['budget'] for row in calibrated['budgets']] == [float(np.quantile(expected_errors, q)) for q in (.25, .5, .75)]
    before = (output / 'calibration.json').read_bytes()
    assert batch.calibrate(output, protocol, lossless) == calibrated
    assert (output / 'calibration.json').read_bytes() == before
    path = snapshots / 'round-01-client-0.npz'
    path.write_bytes(b'corrupted')
    with pytest.raises(RuntimeError, match='checksum'):
        batch.calibrate(output, protocol, lossless)
