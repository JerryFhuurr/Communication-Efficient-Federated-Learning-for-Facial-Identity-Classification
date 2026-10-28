import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from flower_face import refine_finetune as study


@pytest.mark.parametrize('accuracy,loss,count',[(0.834,0.60,1),(0.84,0.64,1),(0.84,0.60,3)])
def test_confirmation_requires_both_accuracy_and_loss(tmp_path,monkeypatch,accuracy,loss,count):
    bases=[]
    for seed in (42,43,44):
        base=dict(config={'seed':seed,'learning-rate':0.003,'num-classes':2},
            output=str(tmp_path/f'old-{seed}'/'run'),best={'validation_accuracy':0.833,'validation_loss':0.618},
            metadata={'manifest_sha256':'manifest'},replay=[])
        bases.append(base)
    previous=tmp_path/'previous.json'
    previous.write_text(json.dumps(dict(status='completed',selected_learning_rate=0.003,runs=bases)))
    monkeypatch.setattr(study,'file_hash',lambda *a:'hash')
    monkeypatch.setattr(study,'source_hash',lambda *a:'source')
    monkeypatch.setattr(study,'model_hash',lambda *a:'model')
    monkeypatch.setattr(study,'snapshot',lambda *a:None)
    monkeypatch.setattr(study,'resolve',lambda *a:SimpleNamespace(Net=lambda *b:SimpleNamespace(state_dict=lambda:{})))
    calls=[]
    monkeypatch.setattr(study,'launch',lambda config,**kw:calls.append(config) or 0)
    def collect(folder,cfg):
        if cfg['seed']<10000:
            return next(b for b in bases if b['config']['seed']==cfg['seed'])
        return dict(config=cfg,output=str(folder/'run'),replay=[],
            best={'validation_accuracy':accuracy,'validation_loss':loss},
            metadata={'initial_model_sha256':'model','source_sha256':'source','manifest_sha256':'manifest'})
    monkeypatch.setattr(study,'collect',collect)
    monkeypatch.setattr(study,'report',lambda *a:None)
    study.run_batch(previous,tmp_path/'output')
    assert len(calls)==count
    assert [c['seed'] for c in calls]==([10042] if count==1 else [10042,10043,10044])
    assert all(c['num-server-rounds']==5 and not c['evaluate-final-test'] for c in calls)
    assert all(c['initial-checkpoint-sha256']=='hash' for c in calls)
    # Re-running a complete batch must not submit additional work.
    study.run_batch(previous,tmp_path/'output',resume=True)
    assert len(calls)==count
