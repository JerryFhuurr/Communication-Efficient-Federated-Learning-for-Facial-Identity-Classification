import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from flwr.app import Context, RecordDict

from flower_face import client_app, server_app
from flower_face.reproducibility import model_hash
from flower_face.task_api import fixed_state_keys, verify_fixed_state


class TinyNet(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([1.0,2.0,3.0]))
        self.register_buffer('running_mean',torch.tensor([0.25,-0.5]))
        self.register_buffer('num_batches_tracked',torch.tensor(0,dtype=torch.int64))


def test_policy_rejects_parameters_and_changed_buffers():
    model=TinyNet(2)
    with pytest.raises(ValueError,match='registered buffers'):
        fixed_state_keys(SimpleNamespace(fixed_state_names=lambda m:['weight']),model)
    with pytest.raises(ValueError,match='Fixed buffer changed'):
        verify_fixed_state({'counter':torch.tensor(1)}, {'counter':torch.tensor(0)})


def test_real_handlers_preserve_buffers_and_replay_lossless_llz(tmp_path,monkeypatch):
    path=tmp_path/'manifest.json'
    manifest=dict(identities=[0,1],num_clients=2,image_variant='fixed-buffer-test',examples=[])
    path.write_text(json.dumps(manifest))
    def train(model, loader, epochs, lr, **kw):
        with torch.no_grad():
            model.weight.add_(torch.tensor([0.02,-0.03,0.007])*(loader.client_id+1))
        return 0.5
    task=SimpleNamespace(Net=TinyNet, fixed_state_names=lambda m:set(dict(m.named_buffers())),
        read_manifest=lambda *a:manifest,
        load_data=lambda cfg,cid,split,*a:SimpleNamespace(client_id=cid,dataset=range(cid+1)),
        train=train,test=lambda model,loader:(float(model.weight.sum()),0.5))
    monkeypatch.setattr(client_app,'resolve',lambda *a,**kw:task)
    monkeypatch.setattr(server_app,'resolve',lambda *a,**kw:task)
    original=TinyNet(2).state_dict()
    results={}
    for method,p,label in [('none',0,'none'),('qsgd',0,'qsgd'),('qsgd-llz',0,'p0'),('qsgd-llz',1,'p1'),('topk-ef',0,'topk'),('topk-ef',0,'topk-sparse')]:
        cfg={'manifest':str(path),'num-classes':2,'num-clients':2,'seed':42,
             'output-dir':str(tmp_path/label),'num-server-rounds':2,'learning-rate':0.01,
             'local-epochs':1,'evaluate-final-test':False,'compression':method,
             'qsgd-levels':127,'llz-p':p,'llz-window':128,'topk-k':1 if label=='topk-sparse' else 3}
        states={1:RecordDict(),2:RecordDict()}
        class Grid:
            def get_node_ids(self):
                return [1,2]
            def send_and_receive(self,messages,timeout):
                for message in messages:
                    state=message.content['arrays'].to_torch_state_dict()
                    verify_fixed_state(state,{k:original[k] for k in ('running_mean','num_batches_tracked')})
                    node=message.metadata.dst_node_id
                    context=Context(run_id=1,node_id=node,node_config={'partition-id':node-1,'num-partitions':2},state=states[node],run_config=cfg)
                    reply=client_app.app(message,context)
                    if method in ('qsgd','qsgd-llz') and message.metadata.message_type=='train':
                        assert set(reply.content['qsgd'])=={'weight'}
                        assert reply.content['metrics']['qsgd_coordinate_count']==3
                    yield reply
        server_app.main(Grid(),Context(run_id=1,node_id=0,node_config={},state=RecordDict(),run_config=cfg))
        folder=next((tmp_path/label).iterdir())
        saved=torch.load(folder/'final_model.pt',weights_only=True)['state_dict']
        verify_fixed_state(saved,{k:original[k] for k in ('running_mean','num_batches_tracked')})
        results[label]=saved
    assert model_hash(results['qsgd'])==model_hash(results['p0'])

    torch.testing.assert_close(results['topk']['weight'],results['none']['weight'])

    torch.testing.assert_close(results['topk-sparse']['weight'],torch.tensor([1.+.04*5/3,2.-.03*5/3,3.]))
