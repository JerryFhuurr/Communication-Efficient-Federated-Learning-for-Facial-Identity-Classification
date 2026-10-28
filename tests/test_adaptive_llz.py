import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.app import ConfigRecord, Context, Message, MetricRecord, RecordDict

from compression import adaptive_llz, qsgd, qsgd_llz
from federated_compression import compression_settings, encode_update
from federated_compression import adaptive_llz as adapter
from flower_face import client_app, server_app
from flower_face.communication import measure_message
from flower_face.task_api import verify_fixed_state
from test_qsgd_flower import wire_round_trip


def values():
    return {'b': np.random.default_rng(9).normal(size=512).astype(np.float64),
            'a': np.random.default_rng(8).normal(size=2048).astype(np.float32)}


@pytest.mark.parametrize('budget', [0., 0.001, 0.05, 1.])
def test_same_quantization_draw_and_whole_update_constraint(budget, monkeypatch):
    inputs = values()
    calls = []
    original = qsgd.quantize
    def quantize(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)
    monkeypatch.setattr(qsgd, 'quantize', quantize)
    rng = np.random.default_rng(3)
    selection = adaptive_llz.select(inputs, levels=127, rng=rng, budget=budget)
    assert len(calls) == len(inputs)  # Not one draw per candidate.
    baseline_rng = np.random.default_rng(3)
    originals = {name: original(inputs[name], levels=127, rng=baseline_rng) for name in sorted(inputs)}
    assert rng.bit_generator.state == baseline_rng.bit_generator.state
    norm = sum(float(np.sum(x.astype(np.float64)**2)) for x in inputs.values())
    for c in selection.candidates:
        extra = total = 0.
        for name, x in inputs.items():
            z = qsgd.dequantize(originals[name]).astype(np.float64)
            restored = qsgd_llz.decode(c.packets[name]).astype(np.float64)
            extra += float(np.sum((restored-z)**2))
            total += float(np.sum((restored-x.astype(np.float64))**2))
            if c.p == 0:
                assert restored.astype(x.dtype).tobytes() == qsgd.dequantize(originals[name]).tobytes()
        assert c.stats['llz_reconstruction_squared_error'] == pytest.approx(extra)
        assert c.stats['llz_total_squared_error'] == pytest.approx(total)
        assert c.stats['qsgd_input_squared_norm'] == pytest.approx(norm)
    feasible = [c for c in selection.candidates if c.stats['llz_reconstruction_squared_error']/norm <= budget]
    assert selection.selected.p == min(feasible, key=lambda c:(c.payload_bytes,c.p)).p
    assert selection.selected.stats['llz_reconstruction_squared_error']/norm <= budget
    if budget == 0:
        assert selection.selected.stats['llz_reconstruction_squared_error'] == 0
    if budget == 1:
        assert selection.selected.p > 0


def test_cost_callback_controls_choice_and_ties_use_lower_p():
    kwargs = dict(levels=127, budget=100.)
    # Deliberately nonmonotone message costs: never assume largest p wins.
    selection = adaptive_llz.select(values(), rng=np.random.default_rng(1),
        cost=lambda c, all_candidates: (300, 100, 200)[c.p], **kwargs)
    assert selection.selected.p == 1
    selection = adaptive_llz.select(values(), rng=np.random.default_rng(1),
        cost=lambda c, all_candidates: 100, **kwargs)
    assert selection.selected.p == 0


def test_zero_norm_and_exact_positive_p_are_feasible_without_epsilon():
    zero = adaptive_llz.select({'z': np.zeros(300, np.float32)}, levels=127,
        rng=np.random.default_rng(1), budget=0., cost=lambda c, _: 100-c.p)
    assert zero.selected.p == 2  # Budget zero does not force p=0.
    for c in zero.candidates:
        assert c.stats['qsgd_input_squared_norm'] == c.stats['llz_reconstruction_squared_error'] == 0
        assert not np.any(qsgd_llz.decode(c.packets['z']))
    assert not adaptive_llz.admissible(1.e-300, 0., 1.)
    # Division can underflow even when both squared measurements are valid.
    assert not adaptive_llz.admissible(1.e-300, 1.e300, 0.)


@pytest.mark.parametrize('budget', [-1., float('nan'), float('inf'), True, '0'])
def test_bad_budget_fails_before_quantizing(budget):
    with pytest.raises(ValueError, match='budget'):
        adaptive_llz.select(values(), levels=127, rng=np.random.default_rng(1), budget=budget)
    with pytest.raises(ValueError, match='budget'):
        compression_settings({'compression':adapter.METHOD,'llz-budget':budget})


@pytest.mark.parametrize('updates', [{}, {'a':np.array([],np.float32)},
    {'a':np.array([np.inf],np.float32)}, {'a':np.ones(3,np.int32)}])
def test_invalid_update(updates):
    with pytest.raises(ValueError):
        adaptive_llz.select(updates, levels=127, rng=np.random.default_rng(1), budget=0.)


def test_tiny_float64_update_cannot_masquerade_as_exact_zero_error():
    with pytest.raises(ValueError,match='scale'):
        adaptive_llz.select({'tiny':np.full(512,1.e-200)},levels=127,
            rng=np.random.default_rng(1),budget=0.)


def reply(budget=0., window=128):
    inputs = values()
    reference = {k:torch.zeros_like(torch.from_numpy(v)) for k,v in inputs.items()}
    trained = {k:torch.from_numpy(v) for k,v in inputs.items()}
    instruction = Message(RecordDict(), dst_node_id=1, message_type='train')
    message = adapter.encode_reply(trained, reference, levels=127, seed=42,
        server_round=1, client_id=0, budget=budget, window=window,
        metrics={'train_loss':1.,'num-examples':10}, instruction=instruction, measure=measure_message)
    return message, reference, trained


@pytest.mark.parametrize('budget', [0., 0.005, 1.])
def test_actual_flower_serialization_preserves_choice_and_accounts_every_byte(budget):
    message, reference, trained = reply(budget)
    wire = wire_round_trip(message)
    assert measure_message(wire) == measure_message(message)
    stats = wire.content['metrics']
    assert stats['adaptive_selected_message_bytes'] == measure_message(wire)['serialized_object_bytes']
    p = wire.content['update']['llz-p']
    assert stats['adaptive_candidate_payload_bytes'][p] == measure_message(wire)['codec_payload_bytes']
    assert stats['adaptive_selected_message_bytes'] > stats['adaptive_candidate_payload_bytes'][p]
    result = adapter.decode(wire.content, reference, levels=127, server_round=1,budget=budget,window=128)
    if budget == 0:
        dense = encode_update(trained, reference, levels=127, seed=42,server_round=1,client_id=0)
        for name in reference:
            assert result[name].numpy().tobytes() == qsgd.decode(dense[name]).tobytes()
    else:
        assert stats['llz_reconstruction_squared_error']/stats['qsgd_input_squared_norm'] <= budget
    again, _, _ = reply(budget)
    assert dict(again.content['qsgd']) == dict(message.content['qsgd'])


@pytest.mark.parametrize('corruption', ['round','policy','p','float-p','budget','inner-p','names',
    'missing-metric','nan','count','payload-cost','candidate-cost','violate-budget','truncated'])
def test_server_rejects_corrupt_or_inconsistent_adaptive_upload(corruption):
    message, reference, trained = reply()
    content = message.content
    if corruption in ('round','policy','p','float-p','budget'):
        key, val = {'round':('server-round',2),'policy':('distortion','total-error'),
            'p':('llz-p',3),'float-p':('llz-p',0.),'budget':('llz-budget',1.)}[corruption]
        content['update'][key] = val
    elif corruption == 'inner-p':
        content['qsgd']['a'] = qsgd_llz.encode(trained['a'].numpy(), levels=127,
            rng=np.random.default_rng(1), p=1)
    elif corruption == 'names':
        content['qsgd']['wrong'] = content['qsgd'].pop('a')
    elif corruption == 'missing-metric':
        del content['metrics']['adaptive_candidate_squared_error']
    elif corruption == 'nan':
        content['metrics']['adaptive_candidate_squared_error'] = [0.,float('nan'),0.]
    elif corruption == 'count':
        content['metrics']['num-examples'] = 0
    elif corruption == 'payload-cost':
        content['metrics']['adaptive_candidate_payload_bytes'] = [0.,0.,0.]
    elif corruption == 'candidate-cost':
        content['metrics']['adaptive_candidate_message_bytes'] = [0.,0.,0.]
    elif corruption == 'violate-budget':
        content['metrics']['llz_reconstruction_squared_error'] = 1.
    else:
        content['qsgd']['a'] = content['qsgd']['a'][:-1]
    with pytest.raises(ValueError):
        adapter.decode(content,reference,levels=127,server_round=1,budget=0.,window=128)


def test_two_round_real_handlers_strategy_checkpoint_and_fixed_buffer_replay(tmp_path,monkeypatch):
    from flower_face.check_adaptive_llz import audit
    from flower_face.study import collect
    class Model(torch.nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.linspace(-.2,.3,128))
            self.register_buffer('counter',torch.tensor(0,dtype=torch.int64))
    manifest = dict(identities=[0,1],num_clients=4,image_variant='adaptive-integration',examples=[])
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    def train(model,loader,*args,**kwargs):
        with torch.no_grad():
            model.weight.add_(torch.linspace(-.02,.03,128)*(loader.client_id+1))
        return float(model.weight.square().mean())
    task=SimpleNamespace(Net=Model,fixed_state_names=lambda model:{'counter'},
        read_manifest=lambda *a:manifest,
        load_data=lambda cfg,cid,split,*args:SimpleNamespace(client_id=cid,dataset=range(cid+1)),
        train=train,test=lambda model,loader:(float(model.weight.square().mean()),.5))
    monkeypatch.setattr(client_app,'resolve',lambda *a,**kw:task)
    monkeypatch.setattr(server_app,'resolve',lambda *a,**kw:task)
    results = {}
    collected = []
    for label,method,budget in [('qsgd','qsgd',0.),('zero',adapter.METHOD,0.),('positive',adapter.METHOD,1.)]:
        cfg={'manifest':str(path),'num-classes':2,'num-clients':4,'seed':42,
            'output-dir':str(tmp_path/label),'num-server-rounds':2,'learning-rate':.01,
            'local-epochs':1,'evaluate-final-test':False,'compression':method,
            'qsgd-levels':127,'llz-budget':budget,'llz-window':128}
        class Grid:
            def get_node_ids(self): return [1,2,3,4]
            def send_and_receive(self,messages,timeout):
                for message in messages:
                    node=message.metadata.dst_node_id
                    context=Context(run_id=1,node_id=node,node_config={'partition-id':node-1,'num-partitions':4},state=RecordDict(),run_config=cfg)
                    yield wire_round_trip(client_app.app(wire_round_trip(message),context))
        server_app.main(Grid(),Context(run_id=1,node_id=0,node_config={},state=RecordDict(),run_config=cfg))
        output=next((tmp_path/label).iterdir())
        final=torch.load(output/'final_model.pt',weights_only=True)['state_dict']
        verify_fixed_state(final,{'counter':torch.tensor(0,dtype=torch.int64)})
        history=[json.loads(line) for line in (output/'history.jsonl').read_text().splitlines()]
        results[label]=dict(final=final,history=history,best=json.loads((output/'best_checkpoint.json').read_text()))
        collected.append(collect(tmp_path/label,cfg))
        if method == adapter.METHOD:
            rows=[json.loads(line) for line in (output/'adaptive_selection.jsonl').read_text().splitlines()]
            assert len(rows)==8 and {r['client'] for r in rows}==set(range(4))
            assert all(r['llz_reconstruction_squared_error']/r['qsgd_input_squared_norm']<=budget for r in rows)
            assert all(r['observed_message_bytes']==r['adaptive_selected_message_bytes'] for r in rows)
            ledger=[json.loads(line) for line in (output/'communication_messages.jsonl').read_text().splitlines()]
            uploads={(r['round'],r['client_partition_id']):r for r in ledger if r['phase']=='train' and r['direction']=='uplink'}
            assert all(r['observed_message_bytes']==uploads[r['round'],r['client']]['serialized_object_bytes'] for r in rows)
    for a,b in zip(results['qsgd']['history'],results['zero']['history']):
        assert a['model_sha256']==b['model_sha256'] and a['validation']==b['validation']
        assert a['train']['train_loss']==b['train']['train_loss']
    assert results['qsgd']['best']==results['zero']['best']
    assert torch.equal(results['qsgd']['final']['weight'],results['zero']['final']['weight'])
    assert audit(collected)['exact_zero_budget_replay']
