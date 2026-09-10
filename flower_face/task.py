"""PyTorch model, local CelebA dataset, and train/evaluation loops (no Flower)."""

import json
import math
import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Net(nn.Module):
    """Small CNN trained from scratch for closed-set identity classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 64),
                                        nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def read_manifest(path, num_classes=10, num_clients=4):
    data = json.loads(Path(path).read_text())
    if len(data["identities"]) != num_classes or data["num_clients"] != num_clients:
        raise ValueError("Manifest classes/clients differ from pyproject.toml configuration.")
    rows = data["examples"]
    if len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Manifest contains overlapping or duplicate images.")
    for row in rows:
        if (not 0 <= row["label"] < num_classes
                or data["identities"][row["label"]] != row["identity"]):
            raise ValueError("Invalid identity label mapping.")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown data split.")
        if row["split"] != "test" and row["client_id"] not in range(num_clients):
            raise ValueError("Invalid client assignment.")
    return data


def validate_augmentation(value):
    if value not in {"none", "horizontal-flip"}:
        raise ValueError("augmentation must be 'none' or 'horizontal-flip'.")
    return value


class CelebASubset(Dataset):
    def __init__(self, manifest, split, client_id=None, image_size=64, augmentation="none"):
        self.root = Path(manifest["image_root"])
        self.rows = [r for r in manifest["examples"] if r["split"] == split
                     and (client_id is None or r["client_id"] == client_id)]
        if not self.rows:
            raise ValueError(f"Empty {split} split for client {client_id}.")
        augmentation = validate_augmentation(augmentation)
        transforms = [Resize((image_size, image_size))]
        if split == "train" and augmentation == "horizontal-flip":
            transforms.append(RandomHorizontalFlip(p=0.5))
        transforms.extend([ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])
        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if 'sha256' in row and hashlib.sha256((self.root / row['filename']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f"Image changed since manifest preparation: {row['filename']}")
        with Image.open(self.root / row["filename"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row["label"]


def load_data(config, client_id=None, split="train", seed=None, *, apply_augmentation=True):
    manifest = read_manifest(config["manifest"], config["num-classes"], config["num-clients"])
    augmentation = validate_augmentation(config.get("augmentation", "none"))
    dataset = CelebASubset(manifest, split, client_id, config["image-size"],
                           augmentation if apply_augmentation else "none")
    generator = torch.Generator().manual_seed(config["seed"] if seed is None else seed)
    return DataLoader(dataset, batch_size=config["batch-size"], shuffle=split == "train",
                      num_workers=0, generator=generator)


def validate_weight_decay(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("weight-decay must be a finite nonnegative number.")
    return float(value)


def validate_config(config):
    """Validate configuration owned by the image-classification task."""
    validate_weight_decay(config.get("weight-decay", 0.0))
    validate_augmentation(config.get("augmentation", "none"))
    image_size = config.get("image-size")
    if type(image_size) is not int or image_size < 8:
        raise ValueError("image-size must be an integer of at least 8")


def train(model, loader, epochs, lr, device="cpu", *, weight_decay=0.0):
    if epochs < 1 or lr <= 0:
        raise ValueError("local-epochs and learning-rate must be positive.")
    model.to(device).train()
    # PyTorch SGD applies coupled L2 decay to every parameter, including biases.
    # Reported loss remains cross-entropy; no penalty is added to evaluation loss.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=validate_weight_decay(weight_decay))
    criterion = nn.CrossEntropyLoss()
    total_loss, count = 0.0, 0
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            count += len(labels)
    return total_loss / count


@torch.no_grad()
def test(model, loader, device="cpu"):
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return loss / count, correct / count
