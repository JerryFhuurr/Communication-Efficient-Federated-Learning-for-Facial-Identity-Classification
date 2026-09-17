"""Federate ResNet-18 layer4 and a head over cached, frozen layer3 outputs."""
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from flower_face.pretrained_task import Features
from flower_face.task import read_manifest as read_image_manifest, test
from flower_face.task import validate_augmentation, validate_weight_decay

# CPU convolutional clients need longer than the small-head task's 120 seconds.
ROUND_TIMEOUT = 900


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=16)
def _verified(path, size, modified, expected):
    if file_hash(path) != expected:
        raise ValueError(f'Artifact checksum mismatch: {path}')


def verify(path, expected):
    path = Path(path).resolve()
    stat = path.stat()
    _verified(str(path), stat.st_size, stat.st_mtime_ns, expected)
    return path


def read_manifest(path, num_classes=100, num_clients=4):
    manifest = read_image_manifest(path, num_classes, num_clients)
    meta = manifest.get('feature_extractor', {})
    if (meta.get('stage') != 'layer3' or meta.get('shape') != [256, 14, 14]
            or meta.get('weights') != 'ResNet18_Weights.IMAGENET1K_V1'
            or meta.get('views') != ['original', 'horizontal-flip']):
        raise ValueError('Expected verified ResNet-18 layer3 features')
    directory = Path(path).resolve().parent
    file = verify(directory / meta['file'], meta['sha256'])
    features = np.load(file, allow_pickle=False, mmap_mode='r')
    if features.shape != (len(manifest['examples']), 2, 256, 14, 14) or features.dtype != np.float32:
        raise ValueError('Invalid intermediate feature shape or dtype')
    verify(directory / meta['weights_file'], meta['weights_sha256'])
    if 'initial_head' in manifest:
        head = manifest['initial_head']
        verify(directory / head['file'], head['sha256'])
    return manifest


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer4 = resnet18(weights=None).layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, features):
        return self.classifier(self.pool(self.layer4(features)).flatten(1))

    def train(self, mode=True):
        super().train(mode)
        # Preserve pretrained running statistics across small federated batches.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self


def build_model(config, classes):
    manifest = read_manifest(config['manifest'], classes, config['num-clients'])
    directory = Path(config['manifest']).resolve().parent
    model = Net(classes)
    state = torch.load(directory / manifest['feature_extractor']['weights_file'],
                       map_location='cpu', weights_only=True)
    model.layer4.load_state_dict({k.removeprefix('layer4.'): v for k, v in state.items()
                                 if k.startswith('layer4.')})
    if 'initial_head' in manifest:
        saved = torch.load(directory / manifest['initial_head']['file'],
                           map_location='cpu', weights_only=True)
        if saved['identities'] != manifest['identities']:
            raise ValueError('Initial head identity mapping differs')
        model.classifier.load_state_dict({k.removeprefix('classifier.'): v
                                         for k, v in saved['state_dict'].items()})
    return model


def validate_config(config):
    validate_augmentation(config.get('augmentation', 'none'))
    validate_weight_decay(config.get('weight-decay', 0.0))
    if config['image-size'] != 224:
        raise ValueError('Intermediate features require 224px preprocessing')
    if config.get('compression', 'none') != 'none':
        raise ValueError('Fine-tuning pilot supports uncompressed training only; BN buffer compression needs a policy')


def load_data(config, client_id=None, split='train', seed=None, *, apply_augmentation=True):
    manifest = read_manifest(config['manifest'], config['num-classes'], config['num-clients'])
    rows = manifest['examples']
    indices = [i for i, row in enumerate(rows) if row['split'] == split
               and (client_id is None or row['client_id'] == client_id)]
    if not indices:
        raise ValueError('Empty intermediate-feature partition')
    array = np.load(Path(config['manifest']).resolve().parent / manifest['feature_extractor']['file'],
                    mmap_mode='r', allow_pickle=False)
    flip = split == 'train' and apply_augmentation and config.get('augmentation') == 'horizontal-flip'
    return DataLoader(Features(array, rows, indices, flip), batch_size=config['batch-size'],
                      shuffle=split == 'train', num_workers=0,
                      generator=torch.Generator().manual_seed(config['seed'] if seed is None else seed))


def train(model, loader, epochs, lr, device='cpu', *, weight_decay=0.0):
    if epochs < 1 or lr <= 0:
        raise ValueError('epochs and learning rate must be positive')
    model.to(device).train()
    optimizer = torch.optim.SGD([
        {'params': model.layer4.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': 10 * lr},
    ], weight_decay=validate_weight_decay(weight_decay))
    total, count = 0.0, 0
    for _ in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
    return total / count
