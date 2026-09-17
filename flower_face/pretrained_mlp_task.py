"""Frozen ResNet-18 features with a small trainable nonlinear head."""
from torch import nn
from flower_face.pretrained_task import load_data, read_manifest, train, test, validate_config


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                        nn.Linear(256, num_classes))

    def forward(self, features):
        return self.classifier(features)
