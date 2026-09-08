import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.supercore.inflatable.inflatable_object import get_all_nested_objects
from flwr.supercore.inflatable.inflatable_utils import inflate_object_from_contents

from compression.qsgd import encode
from flower_face import client_app, server_app
from flower_face.communication import measure_message
from flower_face.updates import CODEC, decode_update, encode_update


class TinyNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([10.0]))


def wire_round_trip(message):
    contents = {key: obj.deflate() for key, obj in get_all_nested_objects(message).items()}
    return inflate_object_from_contents(message.object_id, contents)


@pytest.mark.parametrize('method', ['none', 'qsgd'])
def test_actual_handlers_and_strategy_weight_deltas_and_refresh_round_reference(tmp_path, monkeypatch, method):
    manifest = dict(identities=[100, 200], num_clients=4, image_root=str(tmp_path),
                    image_variant='synthetic-integration', examples=[
                        dict(filename=f'{i}.jpg', label=0, identity=100, split='train', client_id=i)
                        for i in range(4)])
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    config = {'manifest': str(path), 'num-classes': 2, 'num-clients': 4, 'seed': 42,
              'output-dir': str(tmp_path/'outputs'), 'num-server-rounds': 2,
              'learning-rate': 0.01, 'local-epochs': 1, 'evaluate-final-test': False,
              'compression': method, 'qsgd-levels': 127}
    monkeypatch.setattr(client_app, 'Net', TinyNet)
    monkeypatch.setattr(server_app, 'Net', TinyNet)
    monkeypatch.setattr(client_app, 'load_data', lambda config, client_id, split, *args:
                        SimpleNamespace(client_id=client_id, dataset=range(client_id+1)))
    changes = [-1.0, 2.0, 3.0, -4.0]

    def local_train(model, loader, epochs, lr):
        with torch.no_grad():
            model.weight.add_(changes[loader.client_id])
        return float(loader.client_id + 1)

    monkeypatch.setattr(client_app, 'train_model', local_train)
    monkeypatch.setattr(client_app, 'test', lambda model, loader: (float(model.weight.item()), 0.5))

    class DispatchGrid:
        def get_node_ids(self):
            return [1, 2, 3, 4]

        def send_and_receive(self, messages, timeout):
            replies = []
            for msg in messages:
                step = msg.content['config']['server-round']
                is_train = msg.metadata.message_type == 'train'
                # Mean update = (-1*1 + 2*2 + 3*3 - 4*4)/10 = -0.4.
                expected = 10.0 - 0.4 * (step - int(is_train))
                assert msg.content['arrays'].to_torch_state_dict()['weight'].item() == pytest.approx(expected)
                assert set(msg.content) == {'arrays', 'config'}  # full precision downlink
                node = msg.metadata.dst_node_id
                context = Context(run_id=1, node_id=node, node_config={'partition-id': node-1, 'num-partitions': 4},
                                  state=RecordDict(), run_config=config)
                reply = client_app.app(wire_round_trip(msg), context)
                if is_train:
                    if method == 'qsgd':
                        assert set(reply.content) == {'qsgd', 'update', 'metrics'}
                        assert measure_message(reply)['codec_payload_bytes'] > 0
                        assert measure_message(reply)['raw_array_bytes'] == 0
                    else:
                        assert set(reply.content) == {'arrays', 'metrics'}
                else:
                    assert set(reply.content) == {'metrics'}
                replies.append(wire_round_trip(reply))
            return replies

    context = Context(run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config=config)
    server_app.main(DispatchGrid(), context)
    output = next((tmp_path/'outputs').iterdir())
    final = torch.load(output/'final_model.pt', weights_only=True)
    assert final['state_dict']['weight'].item() == pytest.approx(9.2)
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['test'] is None and metrics['best_checkpoint']['step'] == 2
    assert metrics['train']['1']['train_loss'] == pytest.approx(3.0)  # sample-weighted metrics
    assert metrics['config']['compression'] == method
    rows = [json.loads(s) for s in (output/'communication_messages.jsonl').read_text().splitlines()]
    assert len(rows) == 32
    counts = metrics['communication']
    assert counts['total']['serialized_object_bytes'] == sum(r['serialized_object_bytes'] for r in rows)
    if method == 'qsgd':
        assert output.name.startswith('fedavg-qsgd-')
        assert counts['train']['uplink']['raw_array_bytes'] == 0
        assert counts['train']['uplink']['codec_payload_bytes'] == sum(r['codec_payload_bytes'] for r in rows)
        assert counts['train']['uplink']['serialized_object_bytes'] > counts['train']['uplink']['codec_payload_bytes']
    assert counts['downlink']['codec_payload_bytes'] == counts['validation']['total']['codec_payload_bytes'] == 0


def update_content(reference, trained, levels=127, server_round=1):
    return RecordDict({'qsgd': encode_update(trained, reference, levels=levels, seed=42,
                                            server_round=server_round, client_id=0),
                       'update': ConfigRecord({'codec': CODEC, 'levels': levels, 'server-round': server_round}),
                       'metrics': MetricRecord({'num-examples': 50, 'train_loss': 1.0})})


def test_codec_upload_is_smaller_and_meter_counts_packet_bytes_before_decoding():
    reference = {'weight': torch.zeros(4096)}
    trained = {'weight': torch.from_numpy(np.random.default_rng(9).normal(size=4096).astype(np.float32))}
    instruction = Message(RecordDict(), dst_node_id=1, message_type='train')
    compressed = Message(update_content(reference, trained), reply_to=instruction)
    original = Message(RecordDict({'arrays': ArrayRecord(trained),
                                  'metrics': MetricRecord({'num-examples': 50, 'train_loss': 1.0})}), reply_to=instruction)
    small, full = measure_message(compressed), measure_message(original)
    assert small['codec_payload_bytes'] == 4116
    assert small['serialized_object_bytes'] < full['serialized_object_bytes'] / 3
    assert 'Array' not in small['objects_by_type']
    decoded = decode_update(wire_round_trip(compressed).content, reference, levels=127, server_round=1)
    assert decoded['weight'].shape == (4096,) and torch.isfinite(decoded['weight']).all()
    assert measure_message(compressed) == small  # decoded data never replaces measured payload


@pytest.mark.parametrize('corruption', ['round', 'codec', 'levels', 'names', 'shape', 'dtype', 'packet-levels', 'count', 'loss', 'raw'])
def test_server_rejects_incompatible_or_invalid_updates(corruption):
    reference = {'weight': torch.zeros(2)}
    content = update_content(reference, {'weight': torch.ones(2)})
    if corruption in ('round', 'codec', 'levels'):
        key = {'round': 'server-round', 'codec': 'codec', 'levels': 'levels'}[corruption]
        content['update'][key] = 'wrong' if corruption == 'codec' else 9
    elif corruption == 'names':
        content['qsgd']['unexpected'] = content['qsgd'].pop('weight')
    elif corruption in ('shape', 'dtype', 'packet-levels'):
        array = np.ones(3 if corruption == 'shape' else 2, dtype=np.float64 if corruption == 'dtype' else np.float32)
        content['qsgd']['weight'] = encode(array, levels=7 if corruption == 'packet-levels' else 127, rng=np.random.default_rng(1))
    elif corruption == 'count':
        content['metrics']['num-examples'] = 0
    elif corruption == 'loss':
        content['metrics']['train_loss'] = float('nan')
    else:
        content['arrays'] = ArrayRecord(reference)
    with pytest.raises(ValueError):
        decode_update(content, reference, levels=127, server_round=1)


def test_codec_rng_is_separate_and_differs_by_round_and_partition():
    reference = {'weight': torch.zeros(1000)}
    trained = {'weight': torch.linspace(-1, 1, 1000)}
    before_torch = torch.random.get_rng_state().clone()
    before_numpy = np.random.get_state()
    def packet(step, client):
        return encode_update(trained, reference, levels=7, seed=42, server_round=step, client_id=client)['weight']
    first = packet(1, 0)
    assert first == packet(1, 0)
    assert first != packet(1, 1) and first != packet(2, 0)
    assert torch.equal(torch.random.get_rng_state(), before_torch)
    after_numpy = np.random.get_state()
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
