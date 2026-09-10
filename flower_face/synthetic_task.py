"""Small deterministic vector classification task; no image files required."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from flower_face.task import read_manifest, train, test, validate_weight_decay


def validate_config(config):
    """Validate settings used by the vector task."""
    validate_weight_decay(config.get('weight-decay', 0.0))


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, num_classes))

    def forward(self, x):
        return self.layers(x)


def load_data(config, client_id=None, split='train', seed=None, **kwargs):
    manifest = read_manifest(config['manifest'], config['num-classes'], config['num-clients'])
    rows = [r for r in manifest['examples'] if r['split'] == split
            and (client_id is None or r['client_id'] == client_id)]
    if not rows:
        raise ValueError('Empty synthetic partition')
    x = torch.tensor([r['features'] for r in rows], dtype=torch.float32)
    y = torch.tensor([r['label'] for r in rows], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=config['batch-size'],
                      shuffle=split == 'train', num_workers=0,
                      generator=torch.Generator().manual_seed(config['seed'] if seed is None else seed))
