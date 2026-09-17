import json
import pytest
from flower_face import finetune_study as study


@pytest.mark.parametrize('improved, expected_count',[(True,4),(False,2)])
def test_batch_selects_loss_and_only_runs_authorized_confirmation(tmp_path,monkeypatch,improved,expected_count):
    baseline=tmp_path/'baseline'
    baseline.mkdir()
    cfg={'seed':42,'compression':'none','task-module':'flower_face.pretrained_task'}
    (baseline/'metrics.json').write_text(json.dumps(dict(config=cfg,test=None,manifest_sha256='hash',
        best_checkpoint={'validation_loss':1.0,'validation_accuracy':0.7})))
    manifest={'identities':['a'],'num_clients':1,'initial_head':{'sha256':'hash'},
              'feature_extractor':{'source_feature_manifest_sha256':'hash'}}
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(study,'read_manifest',lambda *a:manifest)
    monkeypatch.setattr(study,'file_hash',lambda *a:'hash')
    monkeypatch.setattr(study,'source_hash',lambda *a:'source')
    monkeypatch.setattr(study,'snapshot',lambda *a:None)
    calls=[]
    monkeypatch.setattr(study,'launch',lambda config,**kw: calls.append(config) or 0)
    def collect(folder, config):
        better=config['learning-rate']==0.003
        return dict(output=str(folder),config=config,replay=[],
            metadata=dict(source_sha256='source',manifest_sha256='hash',initial_model_sha256='same'),
            best=dict(validation_loss=(0.8 if improved else 1.1) if better else 1.2,
                      validation_accuracy=0.6 if better else 0.9))
    monkeypatch.setattr(study,'collect',collect)
    monkeypatch.setattr(study,'report',lambda *a:None)
    output=tmp_path/'output'
    study.run_batch(path,output,baseline)
    record=json.loads((output/'study.json').read_text())
    assert record['status']=='completed'
    assert record['selected_learning_rate']==0.003
    assert len(calls)==expected_count
    assert [c['seed'] for c in calls] == ([42,42,43,44] if improved else [42,42])
    assert all(c['compression']=='none' for c in calls)
    # Simulate an interruption after the first completed result; resume must not train it again.
    record['status']='failed'
    record['error']='runtime interruption'
    record['runs']=record['runs'][:1]
    for attempt in record['attempts'][1:]:
        attempt['status']='interrupted'
    (output/'study.json').write_text(json.dumps(record))
    calls.clear()
    study.run_batch(path,output,baseline,resume=True)
    assert len(calls)==expected_count-1
    assert not any(c['seed']==42 and c['learning-rate']==0.001 for c in calls)
