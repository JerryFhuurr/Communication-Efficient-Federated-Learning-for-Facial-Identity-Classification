import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.supercore.inflatable.inflatable_object import get_all_nested_objects
from flwr.supercore.inflatable.inflatable_utils import inflate_object_from_contents

from compression.qsgd import decode as decode_qsgd, encode
from compression import qsgd_llz
from flower_face import client_app, server_app
from flower_face.communication import measure_message
from flower_face.server_app import aggregate_llz_distortion, aggregate_qsgd_distortion
from flower_face.updates import (CODEC, compression_settings, decode_update,
                                 encode_update, encode_update_with_stats,
                                 update_metadata)


class TinyNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([10.0]))


def wire_round_trip(message):
    contents = {key: obj.deflate() for key, obj in get_all_nested_objects(message).items()}
    return inflate_object_from_contents(message.object_id, contents)


@pytest.mark.parametrize('method', ['none', 'qsgd', 'qsgd-llz'])
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
              'compression': method, 'qsgd-levels': 127, 'llz-p': 0, 'llz-window': 128, 'weight-decay': .001}
    monkeypatch.setattr(client_app, 'Net', TinyNet)
    monkeypatch.setattr(server_app, 'Net', TinyNet)
    monkeypatch.setattr(client_app, 'load_data', lambda config, client_id, split, *args:
                        SimpleNamespace(client_id=client_id, dataset=range(client_id+1)))
    changes = [-1.0, 2.0, 3.0, -4.0]

    def local_train(model, loader, epochs, lr, *, weight_decay=0.0):
        assert weight_decay == .001
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
                    if method != 'none':
                        assert set(reply.content) == {'qsgd', 'update', 'metrics', 'client'}
                        assert measure_message(reply)['codec_payload_bytes'] > 0
                        assert measure_message(reply)['raw_array_bytes'] == 0
                    else:
                        assert set(reply.content) == {'arrays', 'metrics', 'client'}
                else:
                    assert set(reply.content) == {'metrics', 'client'}
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
    if method != 'none':
        assert output.name.startswith('fedavg-qsgd-')
        assert counts['train']['uplink']['raw_array_bytes'] == 0
        assert counts['train']['uplink']['codec_payload_bytes'] == sum(r['codec_payload_bytes'] for r in rows)
        assert counts['train']['uplink']['serialized_object_bytes'] > counts['train']['uplink']['codec_payload_bytes']
    assert counts['downlink']['codec_payload_bytes'] == counts['validation']['total']['codec_payload_bytes'] == 0


def update_content(reference, trained, levels=127, server_round=1, method='qsgd', llz_window=128):
    return RecordDict({'qsgd': encode_update(trained, reference, levels=levels, seed=42,
                                            server_round=server_round, client_id=0, method=method, llz_window=llz_window),
                       'update': ConfigRecord(update_metadata(method=method, levels=levels,
                                                              server_round=server_round, llz_window=llz_window)),
                       'metrics': MetricRecord({'num-examples': 50, 'train_loss': 1.0}),
                       'client': ConfigRecord({'partition-id': 0})})


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


@pytest.mark.parametrize('window', [1, 5, 128])
def test_llz_wire_updates_are_exactly_equal_to_qsgd_across_rounds_and_tensors(window):
    generator = torch.Generator().manual_seed(29)
    reference = {'a': torch.randn(13, generator=generator), 'b': torch.randn(7, 9, dtype=torch.float64, generator=generator)}
    for step in [1, 2, 11]:
        trained = {name: base+torch.randn(base.shape, dtype=base.dtype, generator=generator)*0.01
                   for name, base in reference.items()}
        before_torch = torch.random.get_rng_state().clone()
        decoded = []
        for method in ('qsgd', 'qsgd-llz'):
            content = update_content(reference, trained, levels=31, server_round=step, method=method, llz_window=window)
            message = Message(content, dst_node_id=1, message_type='train')
            observed = measure_message(message)
            assert observed['codec_payload_bytes'] == sum(len(packet) for packet in content['qsgd'].values())
            restored = decode_update(wire_round_trip(message).content, reference, levels=31,
                                     server_round=step, method=method, llz_window=window)
            decoded.append(restored)
            assert measure_message(message) == observed
        for name in reference:
            assert decoded[0][name].numpy().tobytes() == decoded[1][name].numpy().tobytes()
        assert torch.equal(torch.random.get_rng_state(), before_torch)
        reference = trained


def test_llz_savings_count_full_messages_with_identical_routing_and_metadata_boundary():
    reference = {'weight': torch.zeros(4096)}
    trained = {'weight': torch.from_numpy(np.random.default_rng(9).normal(size=4096).astype(np.float32))}
    instruction = Message(RecordDict(), dst_node_id=1, message_type='train')
    measurements = []
    for method in ('qsgd', 'qsgd-llz'):
        message = Message(update_content(reference, trained, method=method), reply_to=instruction)
        sizes = measure_message(wire_round_trip(message))
        assert sizes['raw_array_bytes'] == 0 and 'Array' not in sizes['objects_by_type']
        assert sizes['serialized_object_bytes'] > sizes['codec_payload_bytes']
        measurements.append(sizes)
    assert measurements[1]['serialized_object_bytes'] < measurements[0]['serialized_object_bytes']


@pytest.mark.parametrize('corruption', ['codec', 'round', 'outer-p', 'outer-window', 'inner-p', 'inner-window',
    'inner-levels', 'shape', 'dtype', 'plain-packet', 'count', 'missing', 'truncated', 'extra', 'float-metadata'])
def test_llz_integration_rejects_wrong_codec_and_packet_settings(corruption):
    reference = {'weight': torch.zeros(32)}
    content = update_content(reference, {'weight': torch.ones(32)}, method='qsgd-llz')
    if corruption in ('codec', 'round', 'outer-p', 'outer-window', 'float-metadata'):
        key, value = {'codec': ('codec', CODEC), 'round': ('server-round', 2),
                      'outer-p': ('llz-p', 1), 'outer-window': ('llz-window', 64),
                      'float-metadata': ('llz-p', 0.0)}[corruption]
        content['update'][key] = value
    elif corruption in ('inner-p', 'inner-window', 'inner-levels', 'shape', 'dtype'):
        source = np.ones(31 if corruption == 'shape' else 32,
                         dtype=np.float64 if corruption == 'dtype' else np.float32)
        content['qsgd']['weight'] = qsgd_llz.encode(source,
            levels=31 if corruption == 'inner-levels' else 127, rng=np.random.default_rng(1),
            p=1 if corruption == 'inner-p' else 0, window_size=64 if corruption == 'inner-window' else 128)
    elif corruption == 'plain-packet':
        content['qsgd']['weight'] = encode(np.ones(32, dtype=np.float32), levels=127, rng=np.random.default_rng(1))
    elif corruption == 'count':
        content['metrics']['num-examples'] = 0
    elif corruption == 'missing':
        del content['qsgd']['weight']
    elif corruption == 'truncated':
        content['qsgd']['weight'] = content['qsgd']['weight'][:-1]
    else:
        content['arrays'] = ArrayRecord(reference)
    with pytest.raises(ValueError):
        decode_update(content, reference, levels=127, server_round=1, method='qsgd-llz')


@pytest.mark.parametrize('settings', [{'llz-p': -1}, {'llz-p': 255}, {'llz-p': 0.0}, {'llz-p': False},
    {'llz-window': 0}, {'llz-window': True}, {'llz-window': 65536}])
def test_flower_requires_valid_lossy_llz_settings(settings):
    with pytest.raises(ValueError):
        compression_settings(dict(compression='qsgd-llz', **settings))


def test_flower_accepts_lossy_llz_within_qsgd_alphabet():
    assert compression_settings(dict(compression='qsgd-llz', **{
        'qsgd-levels': 7, 'llz-p': 1, 'llz-window': 64})) == ('qsgd-llz', 7)
    assert compression_settings(dict(compression='qsgd-llz', **{
        'qsgd-levels': 7, 'llz-p': 14, 'llz-window': 64})) == ('qsgd-llz', 7)


@pytest.mark.parametrize('p', [0, 1])
def test_llz_secondary_distortion_is_measured_against_same_qsgd_draw(p):
    reference = {'weight': torch.zeros(4096)}
    trained = {'weight': torch.from_numpy(
        np.random.default_rng(12).normal(size=4096).astype(np.float32))}
    packets, stats = encode_update_with_stats(
        trained, reference, levels=127, seed=42, server_round=3,
        client_id=2, method='qsgd-llz', llz_p=p, llz_window=128)
    dense = encode_update(trained, reference, levels=127, seed=42,
                          server_round=3, client_id=2, method='qsgd')
    qsgd_values = qsgd_llz.decode(packets['weight'])
    dense_values = decode_qsgd(dense['weight'])
    error = qsgd_values.astype(np.float64) - dense_values.astype(np.float64)
    assert stats['llz_symbol_count'] == 4096
    assert stats['qsgd_coordinate_count'] == 4096
    assert stats['qsgd_quantization_squared_error'] >= 0
    assert stats['qsgd_input_squared_norm'] > 0
    assert stats['llz_max_code_error'] <= p
    assert stats['llz_reconstruction_squared_error'] == pytest.approx(float(np.sum(error * error)))
    if p == 0:
        assert stats['llz_code_squared_error'] == 0
        assert stats['llz_reconstruction_squared_error'] == 0
    else:
        assert stats['llz_code_squared_error'] > 0
        assert stats['llz_reconstruction_squared_error'] > 0


def test_server_aggregates_llz_distortion_by_coordinates_and_validates_bound():
    def reply(code_error, squared, reference, maximum, symbols):
        return SimpleNamespace(content={'metrics': MetricRecord({
            'llz_code_squared_error': code_error,
            'llz_reconstruction_squared_error': squared,
            'llz_qsgd_squared_norm': reference,
            'llz_max_code_error': maximum,
            'llz_symbol_count': symbols,
        })})
    replies = [reply(3.0, 2.0, 8.0, 1, 10), reply(1.0, 1.0, 4.0, 1, 6)]
    result = aggregate_llz_distortion(replies, p=1)
    assert result['llz_code_mse'] == 0.25
    assert result['llz_relative_squared_error'] == 0.25
    assert result['llz_max_code_error'] == 1
    with pytest.raises(ValueError, match='exceeds configured p'):
        aggregate_llz_distortion([reply(1.0, 1.0, 1.0, 2, 1)], p=1)


def test_server_aggregates_qsgd_distortion_separately():
    def reply(squared, reference, maximum, coordinates):
        return SimpleNamespace(content={'metrics': MetricRecord({
            'qsgd_quantization_squared_error': squared,
            'qsgd_input_squared_norm': reference,
            'qsgd_max_abs_error': maximum,
            'qsgd_coordinate_count': coordinates,
        })})
    result = aggregate_qsgd_distortion(
        [reply(2.0, 8.0, 0.5, 10), reply(1.0, 4.0, 0.25, 5)])
    assert result['qsgd_quantization_mse'] == 0.2
    assert result['qsgd_relative_squared_error'] == 0.25
    assert result['qsgd_max_abs_error'] == 0.5
