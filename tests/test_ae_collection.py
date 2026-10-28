import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from compression.block_data import padded_blocks,normalize_blocks,stratified_counts,snapshot_steps
from flower_face import collect_ae_updates as collection
from flower_face import finetune_task as task
from flower_face.reproducibility import model_hash
from flower_face.task import seed_everything


def test_block_padding_normalization_zero_and_extreme_values():
    x=np.array([3.,4.,0.,0.,0.],np.float32)
    b=padded_blocks(x,4)
    n,s=normalize_blocks(b)
    np.testing.assert_array_equal(s,[5.,0.])
    np.testing.assert_array_equal(n[1],np.zeros(4))
    np.testing.assert_allclose((n*s[:,None]).reshape(-1)[:5],x)
    np.testing.assert_array_equal(b[1,1:],np.zeros(3))
    with pytest.raises(ValueError): padded_blocks(np.array([np.nan],np.float32))
    with pytest.raises(ValueError): padded_blocks(np.ones(3,np.float64))
    with pytest.raises(ValueError): normalize_blocks(np.full((1,256),np.finfo(np.float32).max,np.float32))


@pytest.mark.parametrize('pop,budget',[([1,1,10000],3),([1,3,100],17),([1,3,100],1000),([7,7,7],11)])
def test_sampling_budget_coverage_and_population_weights(pop,budget):
    counts=stratified_counts(pop,budget)
    assert counts.sum()==min(sum(pop),budget)
    assert np.all(counts>=1) and np.all(counts<=pop)
    assert np.sum(counts*(np.asarray(pop)/counts))==pytest.approx(sum(pop))
    np.testing.assert_array_equal(counts,stratified_counts(pop,budget))


def test_snapshot_endpoint_schedule():
    assert snapshot_steps(47)==[5,10,15,19,24,29,33,38,43,47]
    assert snapshot_steps(2)==[1,2]
    assert snapshot_steps(1)==[1]
    with pytest.raises(ValueError): snapshot_steps(0)
    with pytest.raises(ValueError): stratified_counts([1,10],1)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.layer4=nn.Linear(3,2);self.classifier=nn.Linear(2,2)
        self.register_buffer('fixed_counter',torch.tensor(0))
    def forward(self,x): return self.classifier(self.layer4(x))


def loader():
    return DataLoader(TensorDataset(torch.tensor([[1.,2.,3.],[2.,0.,1.],[0.,1.,2.]]),torch.tensor([0,1,0])),batch_size=2)


def test_snapshot_observation_does_not_change_training():
    from copy import deepcopy
    seed_everything(42);a=Tiny();b=deepcopy(a);events=[]
    task.train(a,loader(),1,.003)
    task.train(b,loader(),1,.003,on_batch=lambda step,model:events.append((step,model_hash(model.state_dict()))))
    assert [x[0] for x in events]==[1,2]
    assert model_hash(a.state_dict())==model_hash(b.state_dict())


def test_real_collector_lineage_reproducibility_and_tampering(tmp_path,monkeypatch):
    cfg={'manifest':'unused','num-classes':2,'num-clients':4,'seed':42,'learning-rate':.003,'weight-decay':0.0}
    rows=[{'split':'train','client_id':c,'filename':f'{c}-{i}'} for c in range(4) for i in range(3)]
    rows += [{'split':'test','client_id':0,'filename':'must-not-be-used'}]
    monkeypatch.setattr(task,'read_manifest',lambda *a,**k:{'examples':rows})
    monkeypatch.setattr(task,'build_model',lambda *a,**k:Tiny())
    monkeypatch.setattr(task,'fixed_state_names',lambda model:{'fixed_counter'})
    def load(cfg,cid,split,seed):
        assert split=='train';return loader()
    monkeypatch.setattr(task,'load_data',load)
    seed_everything(42);expected=model_hash(Tiny().state_dict())
    record={'clients':[collection.collect_client(c,cfg,str(tmp_path),expected) for c in range(4)]}
    result=collection.audit(tmp_path,record)
    assert result['status']=='passed' and result['unique_snapshot_ids']==16
    assert result['data_files']==16
    for client in record['clients']:
        fit,val=client['trajectories'];assert fit['seed']!=val['seed']
        assert fit['initial_model_sha256']==val['initial_model_sha256']==expected
        assert not any('full_delta' in x for x in fit['snapshots'])
        assert all('full_delta' in x for x in val['snapshots'])
        with np.load(tmp_path/val['sampled_file']['path']) as sampled:
            for snap in val['snapshots']:
                with np.load(tmp_path/snap['full_delta']['path']) as delta:
                    for stratum in snap['strata']:
                        mask=(sampled['snapshot_step']==snap['step'])&(sampled['tensor_id']==stratum['tensor_id'])
                        raw=padded_blocks(delta[stratum['tensor']])[sampled['block_index'][mask]]
                        np.testing.assert_allclose(sampled['blocks'][mask]*sampled['scales'][mask,None],raw,rtol=1e-6,atol=1e-9)
    again=tmp_path/'again';again.mkdir()
    duplicate=collection.collect_client(0,cfg,str(again),expected)
    with np.load(tmp_path/record['clients'][0]['trajectories'][0]['sampled_file']['path']) as a,np.load(again/duplicate['trajectories'][0]['sampled_file']['path']) as b:
        for key in a.files: np.testing.assert_array_equal(a[key],b[key])
    sample=tmp_path/record['clients'][0]['trajectories'][0]['sampled_file']['path']
    with sample.open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(ValueError,match='checksum'): collection.audit(tmp_path,record)


def test_collection_fingerprint_allows_new_codec_but_detects_training_change(tmp_path):
    import shutil
    root=Path(collection.__file__).resolve().parents[1]
    for package in ('flower_face','compression'):
        (tmp_path/package).mkdir()
        for file in (root/package).glob('*.py'):
            shutil.copy2(file,tmp_path/package/file.name)
    before=collection.collection_source_hash(tmp_path)
    (tmp_path/'compression'/'future_autoencoder.py').write_text('# new independent codec')
    assert collection.collection_source_hash(tmp_path)==before
    file=tmp_path/'flower_face'/'finetune_task.py'
    file.write_text(file.read_text()+'\n# changed training implementation\n')
    assert collection.collection_source_hash(tmp_path)!=before
