import json
import pytest
import torch
from PIL import Image

from flower_face.task import CelebASubset, load_data, validate_augmentation


def fixture_data(tmp_path):
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 0, 255))
    rows = []
    for split in ("train", "validation", "test"):
        filename = f"{split}.png"
        image.save(tmp_path / filename)
        rows.append(dict(filename=filename, label=0, identity=7, split=split,
                         client_id=0 if split != "test" else None))
    manifest = dict(image_root=str(tmp_path), image_variant="synthetic",
                    identities=[7], num_clients=1, examples=rows)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config = {"manifest": str(path), "num-classes": 1, "num-clients": 1,
              "image-size": 2, "batch-size": 1, "seed": 42,
              "augmentation": "horizontal-flip"}
    return manifest, config


def test_horizontal_flip_is_training_only_and_seed_reproducible(tmp_path):
    manifest, _ = fixture_data(tmp_path)
    plain = CelebASubset(manifest, "train", image_size=2, augmentation="none")
    augmented = CelebASubset(manifest, "train", image_size=2, augmentation="horizontal-flip")
    validation = CelebASubset(manifest, "validation", image_size=2,
                              augmentation="horizontal-flip")
    original = plain[0][0]
    torch.manual_seed(0)  # First draw is below 0.5, so torchvision flips.
    first = augmented[0][0]
    torch.manual_seed(0)
    second = augmented[0][0]
    assert torch.equal(first, second)
    assert torch.equal(first, original.flip(-1))
    assert torch.equal(validation[0][0], original)


def test_training_evaluation_can_explicitly_disable_augmentation(tmp_path):
    _, config = fixture_data(tmp_path)
    torch.manual_seed(0)
    augmented = next(iter(load_data(config, split="train")))[0]
    torch.manual_seed(0)
    plain = next(iter(load_data(config, split="train", apply_augmentation=False)))[0]
    assert torch.equal(augmented, plain.flip(-1))


@pytest.mark.parametrize("value", [None, False, "flip", "HorizontalFlip", 1])
def test_invalid_augmentation_is_rejected(value):
    with pytest.raises(ValueError, match="augmentation"):
        validate_augmentation(value)
