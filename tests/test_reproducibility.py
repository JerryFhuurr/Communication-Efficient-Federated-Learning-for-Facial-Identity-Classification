from copy import deepcopy
from itertools import permutations
from types import SimpleNamespace

import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from flower_face.experiment import Experiment
from flower_face.experiment import atomic_replace
from flower_face.reproducibility import model_hash, source_hash
from flower_face.server_app import CheckpointFedAvg, QSGDFedAvg
from flower_face.study import check_replay, snapshot, summarize
from flower_face.task import Net, seed_everything
from federated_compression import CODEC, encode_update_with_stats


def experiment(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    return Experiment(tmp_path/'outputs', 'test', {'manifest': str(manifest)},
                      {'image_variant': 'synthetic-order', 'examples': [], 'identities': [0]}, 'round')


def replies(method, nodes=(400, 200, 300, 100)):
    values = [1e20, -1e20, 3.0, 4.0]
    output = []
    for partition, (value, node) in enumerate(zip(values, nodes)):
        instruction = Message(RecordDict(), dst_node_id=node, message_type='train')
        content = RecordDict({'metrics': MetricRecord({'num-examples': 1, 'train_loss': float(partition)}),
                              'client': ConfigRecord({'partition-id': partition})})
        if method == 'none':
            content['arrays'] = ArrayRecord({'weight': torch.tensor([value])})
        else:
            content['qsgd'], distortion = encode_update_with_stats(
                {'weight': torch.tensor([value])}, {'weight': torch.zeros(1)},
                levels=127, seed=1, server_round=1, client_id=partition)
            content['metrics'].update(distortion)
            content['update'] = ConfigRecord({'codec': CODEC, 'levels': 127, 'server-round': 1})
        output.append(Message(content, reply_to=instruction))
    return output


@pytest.mark.parametrize('method', ['none', 'qsgd'])
def test_aggregation_is_exact_across_arrival_orders_and_runtime_node_ids(tmp_path, method):
    record = experiment(tmp_path)
    strategy = CheckpointFedAvg(record, 4) if method == 'none' else QSGDFedAvg(record, 4, 127)
    if method == 'qsgd':
        strategy.reference = {'weight': torch.zeros(1)}
        strategy.reference_round = 1
    fingerprints = set()
    for nodes in [(400, 200, 300, 100), (15, 17, 19, 13)]:
        strategy.node_partitions = {}
        messages = replies(method, nodes)
        for permutation in permutations(messages):
            arrays, metrics = strategy.aggregate_train(1, permutation)
            fingerprints.add(model_hash(arrays.to_torch_state_dict()))
            assert arrays.to_torch_state_dict()['weight'].item() == 1.75
            assert metrics['train_loss'] == 1.5
    assert len(fingerprints) == 1


def test_original_fedavg_can_change_with_the_same_replies_in_another_order():
    strategy = FedAvg()
    messages = replies('none')
    a, _ = strategy.aggregate_train(1, messages)
    b, _ = strategy.aggregate_train(1, [messages[0], messages[2], messages[1], messages[3]])
    assert a.to_torch_state_dict()['weight'].item() == 1.75
    assert b.to_torch_state_dict()['weight'].item() == 1.0


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'out-of-range', 'changed'])
def test_invalid_partition_identity_is_rejected(tmp_path, kind):
    strategy = CheckpointFedAvg(experiment(tmp_path), 4)
    messages = replies('none')
    if kind == 'missing':
        del messages[0].content['client']
    elif kind == 'duplicate':
        messages[0].content['client']['partition-id'] = 1
    elif kind == 'out-of-range':
        messages[0].content['client']['partition-id'] = 4
    else:
        strategy.checked_replies(messages)
        messages[0].content['client']['partition-id'] = 1
    with pytest.raises(RuntimeError):
        strategy.checked_replies(messages)


def test_seed_and_fingerprint_preserve_shape_dtype_and_values():
    seed_everything(42)
    a = model_hash(Net().state_dict())
    seed_everything(42)
    assert model_hash(Net().state_dict()) == a
    seed_everything(43)
    assert model_hash(Net().state_dict()) != a
    assert torch.are_deterministic_algorithms_enabled()
    assert model_hash({'a': torch.ones(2)}) != model_hash({'a': torch.ones(1,2)})
    assert model_hash({'a': torch.ones(2)}) != model_hash({'a': torch.ones(2, dtype=torch.float64)})


def run_record(seed, method, accuracy):
    return dict(output=f'{seed}-{method}', config={'seed': seed, 'compression': method, 'output-dir': method},
                metadata={k: 'same' for k in ('manifest_sha256','source_sha256','versions','platform',
                           'reproducibility','initial_model_sha256','selection_rule')},
                best={'validation_accuracy': accuracy, 'validation_loss': 2.0}, final_validation={'eval_acc': accuracy},
                total_bytes=100 if method=='none' else 75, train_upload_bytes=40 if method=='none' else 10,
                replay=[{'round':1,'train':{},'validation':{},'model_sha256':'fixed'}])


def test_replay_gate_ignores_transport_sizes_but_catches_model_or_source_changes():
    a=run_record(42,'none',0.4); b=deepcopy(a)
    b['total_bytes']+=1
    check_replay(a,b)
    b['replay'][0]['model_sha256']='changed'
    with pytest.raises(RuntimeError,match='Exact replay'):
        check_replay(a,b)
    b=deepcopy(a); b['metadata']['source_sha256']='changed'
    with pytest.raises(RuntimeError,match='source_sha256'):
        check_replay(a,b)


def test_paired_summary_and_sample_std_reject_missing_duplicate_or_unmatched_runs():
    runs=[run_record(42,'none',0.4),run_record(42,'qsgd',0.5),run_record(43,'none',0.6),run_record(43,'qsgd',0.5)]
    summary=summarize(runs,[42,43])
    assert summary['methods']['none']['selected_accuracy']['mean']==0.5
    assert summary['methods']['none']['selected_accuracy']['sample_std']==pytest.approx(0.1414213562)
    assert summary['paired_accuracy_difference_pp']['mean']==pytest.approx(0)
    assert summary['pairs'][0]['upload_reduction_percent']==75
    for bad in (runs[:-1], runs+[runs[0]]):
        with pytest.raises(RuntimeError):
            summarize(bad,[42,43])
    runs[0]['metadata']['manifest_sha256']='wrong'
    with pytest.raises(RuntimeError,match='manifest_sha256'):
        summarize(runs,[42,43])


def test_snapshot_contains_only_source_and_preserves_fingerprint(tmp_path):
    root=tmp_path/'project'; root.mkdir()
    (root/'pyproject.toml').write_text('[project]\nname="test"')
    for package in ('flower_face','compression'):
        (root/package).mkdir(); (root/package/'__init__.py').write_text('')
    (root/'img_align_celeba').mkdir(); (root/'img_align_celeba'/'image.jpg').write_bytes(b'not copied')
    destination=tmp_path/'snapshot'
    snapshot(root,destination)
    assert not (destination/'img_align_celeba').exists()
    assert source_hash(root)==source_hash(destination)
    (destination/'pyproject.toml').write_text('[project]\nname = "test"\n')
    assert source_hash(root)==source_hash(destination)
    (destination/'pyproject.toml').write_text('[project]\nname = "changed"\n')
    assert source_hash(root)!=source_hash(destination)


def test_atomic_replace_retries_transient_windows_locks(tmp_path, monkeypatch):
    from pathlib import Path
    original = Path.replace
    attempts = []
    def flaky_replace(source, target):
        attempts.append(1)
        if len(attempts) < 3:
            error = PermissionError('temporary reader lock')
            error.winerror = 5
            raise error
        return original(source, target)
    source, target = tmp_path/'temporary', tmp_path/'target'
    source.write_text('new'); target.write_text('old')
    monkeypatch.setattr(Path, 'replace', flaky_replace)
    monkeypatch.setattr('flower_face.experiment.time.sleep', lambda _: None)
    atomic_replace(source, target)
    assert target.read_text() == 'new' and len(attempts) == 3


def test_atomic_replace_does_not_hide_persistent_errors(tmp_path, monkeypatch):
    from pathlib import Path
    attempts = []
    def locked(source, target):
        attempts.append(1)
        error = PermissionError('persistent lock'); error.winerror = 32
        raise error
    monkeypatch.setattr(Path, 'replace', locked)
    monkeypatch.setattr('flower_face.experiment.time.sleep', lambda _: None)
    with pytest.raises(PermissionError):
        atomic_replace(tmp_path/'source', tmp_path/'target')
    assert len(attempts) == 8
