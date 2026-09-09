import json
from copy import deepcopy

import pytest
import torch
from PIL import Image

from flower_face.task import CelebASubset, load_data, validate_augmentation
from scripts.sweep_augmentation import check_control, summarize


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


def sweep_result(seed=42, augmentation="none", loss=2.0):
    return dict(config={"seed": seed, "augmentation": augmentation, "learning-rate": .1,
                        "weight-decay": 0., "compression": "none",
                        "evaluate-final-test": False, "output-dir": "example"},
        metadata=dict.fromkeys(("manifest_sha256", "initial_model_sha256", "versions", "python",
            "platform", "reproducibility", "selection_rule", "source_sha256",
            "communication_measurement"), "same"),
        replay=[dict(round=1, model_sha256="model", train={"loss": 1.}, validation={"loss": 2.})],
        best=dict(validation_loss=loss, validation_accuracy=.4),
        checkpoint_evaluation={"final": {"train": {"accuracy": .8}}})


def test_augmentation_legacy_bridge_requires_exact_disabled_replay():
    old, new = sweep_result(), sweep_result()
    del old["config"]["augmentation"]
    new["metadata"]["source_sha256"] = "augmentation-support"
    new["metadata"]["communication_measurement"] = "new-schema"
    check_control(old, new, legacy_replay=True)
    changed = deepcopy(new)
    changed["replay"][0]["model_sha256"] = "different"
    with pytest.raises(RuntimeError, match="replay"):
        check_control(old, changed, legacy_replay=True)
    new["config"]["augmentation"] = "horizontal-flip"
    with pytest.raises(RuntimeError):
        check_control(old, new, legacy_replay=True)


def test_augmented_comparison_allows_only_augmentation_and_output_changes():
    base, candidate = sweep_result(), sweep_result(augmentation="horizontal-flip")
    check_control(base, candidate)
    for key, value in (("learning-rate", .03), ("weight-decay", .001), ("seed", 43)):
        changed = deepcopy(candidate)
        changed["config"][key] = value
        with pytest.raises(RuntimeError, match="confound"):
            check_control(base, changed)


def test_augmentation_summary_requires_full_matrix_and_uses_mean_loss():
    runs = [sweep_result(seed, method, 2.0-(.1 if method=="horizontal-flip" else 0))
            for seed in (42,43,44) for method in ("none","horizontal-flip")]
    result = summarize(runs, [42,43,44], ["none","horizontal-flip"])
    assert result["preferred_augmentation"] == "horizontal-flip"
    with pytest.raises(RuntimeError):
        summarize(runs[:-1], [42,43,44], ["none","horizontal-flip"])
    for run in runs:
        run["best"]["validation_loss"] = 2.
    assert summarize(runs, [42,43,44], ["none","horizontal-flip"])["preferred_augmentation"] == "none"
