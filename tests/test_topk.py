import struct
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from flwr.app import RecordDict, MetricRecord, ConfigRecord
from compression import topk
from federated_compression import topk as adapter


def test_ties_full_density_and_packet_size():
    x=np.array([2,-2,2,1],dtype=np.float32)
    p=topk.encode(x,2)
    assert len(p)==12+8*2
    np.testing.assert_array_equal(topk.decode(p,expected_size=4,expected_k=2),[2,-2,0,0])
    np.testing.assert_array_equal(topk.decode(topk.encode(x,4),expected_size=4,expected_k=4),x)


@pytest.mark.parametrize('k',[0,-1,5,True,1.5])
def test_bad_k(k):
    with pytest.raises(ValueError): topk.encode(np.ones(4,dtype=np.float32),k)


@pytest.mark.parametrize('x',[np.array([np.nan],dtype=np.float32),np.array([np.inf],dtype=np.float32),np.ones(4,dtype=np.float64),np.ones((2,2),dtype=np.float32)])
def test_invalid_input(x):
    with pytest.raises(ValueError): topk.encode(x,1)


@pytest.mark.parametrize('kind',['truncated','extra','magic','duplicate','out-of-range','nan'])
def test_bad_packets(kind):
    p=bytearray(topk.encode(np.arange(4,dtype=np.float32),2))
    if kind=='truncated': p=p[:-1]
    if kind=='extra': p+=b'x'
    if kind=='magic': p[:4]=b'NOPE'
    if kind=='duplicate': p[16:20]=p[12:16]
    if kind=='out-of-range': p[16:20]=struct.pack('<I',4)
    if kind=='nan': p[20:24]=struct.pack('<f',float('nan'))
    with pytest.raises(ValueError): topk.decode(bytes(p),expected_size=4,expected_k=2)


def test_feedback_conservation_persistence_isolation_and_metadata():
    base={'a':torch.zeros(3)}
    ctx=SimpleNamespace(run_id=1,state=RecordDict())
    total=np.zeros(3); sent=np.zeros(3)
    for r in range(1,4):
        # Reconstruct Context state through serialization, as across worker processes.
        if r>1:
            saved=ctx.state['topk-residual'].to_numpy_ndarrays()
            from flwr.app import ArrayRecord
            ctx=SimpleNamespace(run_id=1,state=RecordDict({'topk-residual':ArrayRecord(saved),
                 'topk-meta':ConfigRecord(dict(ctx.state['topk-meta']))}))
        delta=torch.tensor([1.,0.5,0.25])
        payload,meta,stats=adapter.encode({'a':delta},base,ctx,k=1,server_round=r,client_id=0)
        content=RecordDict({'topk':payload,'update':meta,'metrics':MetricRecord({'num-examples':1,'train_loss':0.5}), 'client':ConfigRecord({'partition-id':0})})
        out=adapter.decode(content,base,k=1,server_round=r)['a'].numpy()
        total+=delta.numpy();sent+=out
        residual=ctx.state['topk-residual'].to_numpy_ndarrays()[0]
        np.testing.assert_array_equal(total,sent+residual)
    with pytest.raises(ValueError,match='stale'):
        adapter.encode(base,base,ctx,k=1,server_round=3,client_id=0)
    with pytest.raises(ValueError,match='stale'):
        adapter.encode(base,base,ctx,k=1,server_round=4,client_id=1)
    with pytest.raises(ValueError,match='Missing'):
        adapter.encode(base,base,SimpleNamespace(run_id=1,state=RecordDict()),k=1,server_round=2,client_id=0)
    with pytest.raises(ValueError,match='metadata'):
        adapter.decode(content,{'b':torch.zeros(3)},k=1,server_round=3)


def test_historical_compatibility_rejects_training_changes():
    from copy import deepcopy
    from flower_face.check_topk import compatibility
    meta={k: 'same' for k in ('manifest_sha256','initial_model_sha256','versions','platform','reproducibility','selection_rule','fixed_state_policy')}
    a={'metadata':meta,'config':{'compression':'qsgd-llz','llz-p':1,'seed':42,'batch-size':16}}
    b=deepcopy(a);b['config'].update(compression='topk-ef',**{'topk-k':124000,'llz-p':0})
    compatibility(a,b)
    b['config']['seed']=43
    with pytest.raises(ValueError,match='configurations'): compatibility(a,b)


def test_global_selection_across_named_tensors_and_wire_accounting():
    from flwr.app import Message
    from flower_face.communication import measure_message
    ctx=SimpleNamespace(run_id=5,state=RecordDict())
    base={'z':torch.zeros((1,2)), 'a':torch.zeros(2)}
    trained={'z':torch.tensor([[3.,-4.]]),'a':torch.tensor([1.,2.])}
    packet,meta,stats=adapter.encode(trained,base,ctx,k=2,server_round=1,client_id=0)
    records=RecordDict({'topk':packet,'update':meta,'metrics':MetricRecord({'train_loss':.1,'num-examples':1}), 'client':ConfigRecord({'partition-id':0})})
    restored=adapter.decode(records,base,k=2,server_round=1)
    torch.testing.assert_close(restored['a'],torch.zeros(2))
    torch.testing.assert_close(restored['z'],trained['z'])
    size=measure_message(Message(content=records,dst_node_id=1,message_type='train'))
    assert size['codec_payload_bytes']==28
    assert size['serialized_object_bytes']>28


def test_platform_mismatch_is_explicit_not_silently_matched():
    from copy import deepcopy
    from flower_face.check_topk import compatibility
    meta={k:'same' for k in ('manifest_sha256','initial_model_sha256','versions','platform','reproducibility','selection_rule','fixed_state_policy')}
    a={'metadata':meta,'config':{}}
    b=deepcopy(a);b['metadata']['platform']='different-build'
    with pytest.raises(ValueError,match='platform'): compatibility(a,b)
    assert compatibility(a,b,allow_platform_difference=True)=={'baseline':'same','candidate':'different-build'}
    b['metadata']['initial_model_sha256']='changed'
    with pytest.raises(ValueError,match='initial_model'): compatibility(a,b,allow_platform_difference=True)


def test_runner_platform_preflight_stops_before_training(tmp_path, monkeypatch):
    import json,sys
    from flower_face import check_topk
    (tmp_path/'study.json').write_text(json.dumps({'status':'completed','runs':[{'metadata':{'platform':'old-build'}}]}))
    monkeypatch.setattr(sys,'argv',['check_topk','--baseline',str(tmp_path)])
    monkeypatch.setattr(check_topk.platform,'platform',lambda:'new-build')
    monkeypatch.setattr(check_topk,'launch',lambda *a,**k:pytest.fail('Training must not launch'))
    with pytest.raises(ValueError,match='no training launched'): check_topk.main()
