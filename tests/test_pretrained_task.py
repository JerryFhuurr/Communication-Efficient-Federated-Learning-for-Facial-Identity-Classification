import hashlib
import json
import numpy as np
import pytest
import torch
from flower_face import pretrained_task as task


def fixture_config(tmp_path):
    features = np.zeros((4, 2, 512), dtype=np.float32)
    features[:, 1, :] = 1
    np.save(tmp_path/'features.npy',features)
    rows=[dict(filename=str(i),identity=str(i%2),label=i%2,split=split,client_id=0)
          for i,split in enumerate(['train','train','validation','validation'])]
    manifest=dict(identities=['0','1'],num_clients=1,examples=rows,
        feature_extractor=dict(weights='ResNet18_Weights.IMAGENET1K_V1',dimension=512,
            file='features.npy',sha256=hashlib.sha256((tmp_path/'features.npy').read_bytes()).hexdigest()))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return {'manifest':str(tmp_path/'manifest.json'),'num-classes':2,'num-clients':1,
            'seed':42,'batch-size':2,'image-size':224,'augmentation':'horizontal-flip'}


def test_validation_uses_original_view_and_head_can_train(tmp_path):
    config=fixture_config(tmp_path)
    validation=task.load_data(config,0,'validation')
    assert torch.count_nonzero(next(iter(validation))[0]) == 0
    training=task.load_data(config,0,'train',apply_augmentation=False)
    assert torch.count_nonzero(next(iter(training))[0]) == 0
    head=task.Net(2)
    before=head.classifier.bias.detach().clone()
    task.train(head,training,1,0.01)
    assert not torch.equal(before,head.classifier.bias)
    assert all(v.is_floating_point() for v in head.state_dict().values())
    assert sum(p.numel() for p in task.Net(100).parameters()) == 51300


def test_modified_feature_cache_is_rejected(tmp_path):
    config=fixture_config(tmp_path)
    with (tmp_path/'features.npy').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='cache changed'):
        task.read_manifest(config['manifest'],2,1)


def test_flip_view_only_used_when_enabled(tmp_path,monkeypatch):
    config=fixture_config(tmp_path)
    monkeypatch.setattr(torch,'rand',lambda *args: torch.tensor(0.0))
    assert torch.all(next(iter(task.load_data(config,0,'train')))[0] == 1)
    assert torch.all(next(iter(task.load_data(config,0,'validation')))[0] == 0)
    with pytest.raises(ValueError,match='224'):
        task.validate_config(dict(config,**{'image-size':64}))
