"""CIFAR-10 classification with a 32-pixel ResNet-18 and GroupNorm."""

from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from flower_face.prepare_cifar10 import ARCHIVE_MD5, CLASSES, COUNTS, RECORD_BYTES, digest
from flower_face.task import read_manifest as read_examples
from flower_face.task import train, test, validate_augmentation, validate_weight_decay


ROUND_TIMEOUT = 3600


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=None, num_classes=num_classes,
                              norm_layer=partial(nn.GroupNorm, 32))
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.model.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.model.maxpool = nn.Identity()

    def forward(self, images):
        return self.model(images)


def validate_config(config):
    if config["image-size"] != 32:
        raise ValueError("CIFAR-10 requires image-size=32")
    validate_augmentation(config.get("augmentation", "none"))
    validate_weight_decay(config.get("weight-decay", 0.0))


def file_signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=8)
def verified_manifest(path, num_classes, num_clients, signatures):
    manifest = read_examples(path, num_classes, num_clients)
    if manifest["identities"] != CLASSES or manifest.get("archive_md5") != ARCHIVE_MD5:
        raise ValueError("Expected a prepared CIFAR-10 manifest")
    for split in COUNTS:
        artifact = path.parent / f"{split}.bin"
        metadata = manifest["cifar10_files"][split]
        if (metadata["records"] != COUNTS[split] or artifact.stat().st_size != COUNTS[split] * RECORD_BYTES
                or digest(artifact) != metadata["sha256"]):
            raise ValueError(f"CIFAR-10 data failed verification: {artifact}")
    for row in manifest["examples"]:
        source = "test" if row["split"] == "test" else "train"
        index = row["index"]
        if (row["source"] != source or type(index) is not int or not 0 <= index < COUNTS[source]
                or row["filename"] != f"{source}:{index}"):
            raise ValueError("Invalid CIFAR-10 sample reference")
    return manifest


def read_manifest(path, num_classes=10, num_clients=4):
    path = Path(path).resolve()
    signatures = tuple(file_signature(file) for file in (path, path.parent / "train.bin", path.parent / "test.bin"))
    return verified_manifest(path, num_classes, num_clients, signatures)


class CIFAR10Records(Dataset):
    def __init__(self, path, source, rows, flip):
        self.records = np.memmap(path, dtype=np.uint8, mode="r", shape=(COUNTS[source], RECORD_BYTES))
        self.indices = np.array([row["index"] for row in rows], dtype=np.int64)
        self.labels = np.array([row["label"] for row in rows], dtype=np.int64)
        if not np.array_equal(self.records[self.indices, 0], self.labels):
            raise ValueError("CIFAR-10 manifest labels differ from the data")
        self.flip = flip

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        pixels = self.records[self.indices[index], 1:].copy()
        image = torch.from_numpy(pixels).reshape(3, 32, 32).float().div_(127.5).sub_(1)
        if self.flip and torch.rand(()).item() < 0.5:
            image = image.flip(-1)
        return image, int(self.labels[index])


def load_data(config, client_id=None, split="train", seed=None, *, apply_augmentation=True):
    path = Path(config["manifest"]).resolve()
    manifest = read_manifest(path, config["num-classes"], config["num-clients"])
    rows = [row for row in manifest["examples"] if row["split"] == split
            and (client_id is None or row["client_id"] == client_id)]
    if not rows:
        raise ValueError(f"Empty CIFAR-10 {split} split for client {client_id}")
    source = "test" if split == "test" else "train"
    flip = split == "train" and apply_augmentation and config.get("augmentation") == "horizontal-flip"
    dataset = CIFAR10Records(path.parent / f"{source}.bin", source, rows, flip)
    generator = torch.Generator().manual_seed(config["seed"] if seed is None else seed)
    return DataLoader(dataset, batch_size=config["batch-size"], shuffle=split == "train",
                      num_workers=0, generator=generator)
