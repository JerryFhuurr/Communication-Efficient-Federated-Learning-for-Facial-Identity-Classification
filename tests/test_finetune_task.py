import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

from flower_face import finetune_task as task
from flower_face.task_api import resolve


def test_split_model_matches_full_resnet_and_only_tail_is_registered():
    torch.set_num_threads(1)
    full = resnet18(weights=None).eval()
    tail = task.Net(3).eval()
    tail.layer4.load_state_dict(full.layer4.state_dict())
    full.fc = tail.classifier
    prefix = nn.Sequential(*list(full.children())[:7])
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(tail(prefix(images)), full(images))
    assert all(k.startswith(('layer4.', 'classifier.')) for k in tail.state_dict())


def test_training_updates_tail_and_head_but_preserves_bn_statistics():
    torch.set_num_threads(1)
    model = task.Net(3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = DataLoader(TensorDataset(torch.rand(4,256,14,14), torch.tensor([0,1,2,0])), batch_size=2)
    task.train(model, loader, 1, 0.001)
    after = model.state_dict()
    assert not torch.equal(before['layer4.0.conv1.weight'], after['layer4.0.conv1.weight'])
    assert not torch.equal(before['classifier.weight'], after['classifier.weight'])
    for key in before:
        if 'running_' in key or 'num_batches_tracked' in key:
            assert torch.equal(before[key], after[key]), key
    assert all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))


@pytest.fixture
def config(tmp_path):
    array = np.zeros((4,2,256,14,14), np.float32)
    array[:,1] = 1
    np.save(tmp_path/'features.npy',array)
    model = resnet18(weights=None)
    torch.save(model.state_dict(), tmp_path/'weights.pt')
    rows = [dict(filename=str(i), identity=str(i%2), label=i%2, split=split, client_id=0)
            for i,split in enumerate(('train','train','validation','validation'))]
    manifest = dict(identities=['0','1'],num_clients=1,examples=rows,
        feature_extractor=dict(stage='layer3',shape=[256,14,14],weights='ResNet18_Weights.IMAGENET1K_V1',
            views=['original','horizontal-flip'],file='features.npy',sha256=task.file_hash(tmp_path/'features.npy'),
            weights_file='weights.pt',weights_sha256=task.file_hash(tmp_path/'weights.pt')))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return {'task-module':'flower_face.finetune_task','manifest':str(tmp_path/'manifest.json'),
            'num-classes':2,'num-clients':1,'batch-size':2,'seed':42,'image-size':224,
            'augmentation':'horizontal-flip','compression':'none'}


def test_factory_initializes_pretrained_tail_and_loader_is_split_safe(config, monkeypatch):
    model = resolve(config).Net(2)
    weights = torch.load(str(config['manifest']).replace('manifest.json','weights.pt'),weights_only=True)
    assert torch.equal(model.layer4[0].conv1.weight,weights['layer4.0.conv1.weight'])
    monkeypatch.setattr(torch,'rand',lambda *a: torch.tensor(0.0))
    assert torch.all(next(iter(task.load_data(config)))[0] == 1)
    assert torch.all(next(iter(task.load_data(config,split='validation')))[0] == 0)
    assert torch.all(next(iter(task.load_data(config,apply_augmentation=False)))[0] == 0)
    file = str(config['manifest']).replace('manifest.json','features.npy')
    with open(file,'ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError,match='checksum'):
        task.load_data(config)


def test_pilot_rejects_compressed_bn_and_wrong_preprocessing():
    with pytest.raises(ValueError,match='uncompressed'):
        task.validate_config({'image-size':224,'compression':'qsgd'})
    with pytest.raises(ValueError,match='224'):
        task.validate_config({'image-size':64})


@pytest.mark.parametrize('module', ['flower_face.pretrained_task', 'flower_face.pretrained_mlp_task',
                                   'flower_face.finetune_task'])
def test_predictor_reconstructs_saved_image_model(config, module):
    from importlib import import_module
    from flower_face.predict_pretrained import load_predictor
    directory = Path(config['manifest']).parent
    manifest = json.loads(Path(config['manifest']).read_text())
    if module != 'flower_face.finetune_task':
        manifest['feature_extractor']['dimension'] = 512
        manifest['feature_extractor'].pop('stage')
    Path(config['manifest']).write_text(json.dumps(manifest))
    head = import_module(module).Net(2).eval()
    saved = dict(manifest_sha256=task.file_hash(config['manifest']),identities=manifest['identities'],
                 config={'task-module':module},state_dict=head.state_dict())
    checkpoint = directory/'model.pt'
    torch.save(saved, checkpoint)
    predictor, _, _ = load_predictor(checkpoint, config['manifest'])
    backbone = resnet18(weights=None).eval()
    backbone.load_state_dict(torch.load(directory/'weights.pt',weights_only=True))
    if module == 'flower_face.finetune_task':
        backbone = nn.Sequential(*list(backbone.children())[:7])
    else:
        backbone.fc = nn.Identity()
    images = torch.rand(2,3,224,224)
    with torch.no_grad():
        assert torch.equal(predictor(images),head(backbone(images)))
    saved['manifest_sha256'] = 'incorrect'
    torch.save(saved,checkpoint)
    with pytest.raises(ValueError,match='do not match'):
        load_predictor(checkpoint,config['manifest'])
