"""Task contract: Net(classes), load_data, train, test and read_manifest.

Task modules belong under flower_face so source snapshots include their code.
The default remains the image classification example for compatibility.
"""
from importlib import import_module
from types import SimpleNamespace
import math
import torch


def resolve(config, **defaults):
    name = config.get('task-module', 'flower_face.task')
    if not isinstance(name, str) or not name.startswith('flower_face.'):
        raise ValueError('task-module must be a module under flower_face')
    if name == 'flower_face.task' and defaults:
        return SimpleNamespace(**defaults)
    module = import_module(name)
    for key in ('Net', 'load_data', 'train', 'test', 'read_manifest'):
        if not callable(getattr(module, key, None)):
            raise ValueError(f'Task {name} is missing {key}')
    return module


def validate(config):
    """Fail before runtime startup for unsupported task/configuration contracts."""
    for key in ('num-classes','num-clients','num-server-rounds','local-epochs','batch-size'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    lr = config.get('learning-rate')
    if isinstance(lr, bool) or not isinstance(lr, (float,int)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('learning-rate must be finite and positive')
    task = resolve(config)
    task.read_manifest(config['manifest'], config['num-classes'], config['num-clients'])
    model = task.Net(config['num-classes'])
    if config.get('compression','none') != 'none':
        for name, value in model.state_dict().items():
            if value.dtype not in (torch.float32, torch.float64):
                raise ValueError(f'Compressed task requires float32/64 state tensors: {name}. '
                                 'Integer buffers need an explicit aggregation policy.')
    for split in ('train','validation'):
        for client in range(config['num-clients']):
            loader = task.load_data(config, client, split)
            if len(loader.dataset) < 1:
                raise ValueError(f'Empty {split} for client {client}')
    return task
