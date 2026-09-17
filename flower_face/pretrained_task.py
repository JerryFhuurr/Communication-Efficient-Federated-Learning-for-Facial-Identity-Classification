"""Frozen ResNet-18 features; only the 512-input linear head is federated."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from flower_face.task import read_manifest as read_image_manifest
from flower_face.task import train, test, validate_augmentation, validate_weight_decay


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, features):
        return self.classifier(features)


def read_manifest(path, num_classes=100, num_clients=4):
    manifest = read_image_manifest(path, num_classes, num_clients)
    meta = manifest.get('feature_extractor', {})
    if meta.get('weights') != 'ResNet18_Weights.IMAGENET1K_V1' or meta.get('dimension') != 512:
        raise ValueError('Expected frozen ResNet-18 IMAGENET1K_V1 features')
    file = Path(path).resolve().parent / meta['file']
    if hashlib.sha256(file.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('Feature cache changed; prepare a new manifest')
    features = np.load(file, allow_pickle=False, mmap_mode='r')
    if features.shape != (len(manifest['examples']), 2, 512) or features.dtype != np.float32:
        raise ValueError('Invalid feature array shape or dtype')
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite features')
    return manifest


def validate_config(config):
    validate_weight_decay(config.get('weight-decay', 0.0))
    validate_augmentation(config.get('augmentation', 'none'))
    if config['image-size'] != 224:
        raise ValueError('Cached ResNet-18 features use fixed 224px preprocessing')


class Features(Dataset):
    def __init__(self, array, rows, indices, flip):
        self.array, self.rows, self.indices, self.flip = array, rows, indices, flip

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        view = int(torch.rand(()).item() < 0.5) if self.flip else 0
        return torch.from_numpy(self.array[i, view].copy()), self.rows[i]['label']


def load_data(config, client_id=None, split='train', seed=None, *, apply_augmentation=True):
    manifest = read_manifest(config['manifest'], config['num-classes'], config['num-clients'])
    rows = manifest['examples']
    indices = [i for i,r in enumerate(rows) if r['split'] == split and
               (client_id is None or r['client_id'] == client_id)]
    if not indices:
        raise ValueError('Empty feature partition')
    array = np.load(Path(config['manifest']).resolve().parent / manifest['feature_extractor']['file'],
                    allow_pickle=False, mmap_mode='r')
    flip = split == 'train' and apply_augmentation and config.get('augmentation') == 'horizontal-flip'
    return DataLoader(Features(array, rows, indices, flip), batch_size=config['batch-size'],
                      shuffle=split == 'train', num_workers=0,
                      generator=torch.Generator().manual_seed(config['seed'] if seed is None else seed))
