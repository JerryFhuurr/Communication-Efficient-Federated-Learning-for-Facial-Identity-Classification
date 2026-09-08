from collections import Counter
import json

import pytest
import torch
from PIL import Image
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict

from flower_face.client_app import app
from flower_face.prepare_data import select_examples
from flower_face.task import Net, read_manifest


@pytest.fixture
def subset(tmp_path):
    annotations = tmp_path / "identity_CelebA.txt"
    annotations.write_text("\n".join(f"{identity * 30 + n:06d}.jpg {identity}"
                                     for identity in range(12) for n in range(30)))
    identities, rows = select_examples(annotations)
    return annotations, identities, rows


def test_reproducible_disjoint_balanced_split(subset):
    annotations, identities, rows = subset
    assert (identities, rows) == select_examples(annotations)
    assert (identities, rows) != select_examples(annotations, seed=43)
    assert len({r["filename"] for r in rows}) == 300
    assert Counter(r["split"] for r in rows) == {"train": 200, "validation": 40, "test": 60}
    for client in range(4):
        train = [r for r in rows if r["client_id"] == client and r["split"] == "train"]
        val = [r for r in rows if r["client_id"] == client and r["split"] == "validation"]
        assert Counter(r["label"] for r in train) == dict.fromkeys(range(10), 5)
        assert Counter(r["label"] for r in val) == dict.fromkeys(range(10), 1)


def test_insufficient_images_fails(subset):
    with pytest.raises(ValueError, match="Only 0 identities"):
        select_examples(subset[0], images_per_identity=40)


def test_manifest_rejects_overlap(tmp_path, subset):
    _, identities, rows = subset
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(identities=identities, num_clients=4, examples=rows + [rows[0]])))
    with pytest.raises(ValueError, match="overlapping"):
        read_manifest(path)


def test_flower_handlers_train_and_validate(tmp_path, subset):
    """Exercise actual Message API dispatch, serialization, optimizer, and metrics."""
    _, identities, rows = subset
    for row in rows:
        if row["client_id"] == 0:
            Image.new("RGB", (178, 218), (row["label"] * 20, 80, 120)).save(tmp_path / row["filename"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(image_root=str(tmp_path), identities=identities,
                                       num_clients=4, examples=rows)))
    config = {"manifest": str(manifest), "num-classes": 10, "num-clients": 4,
              "seed": 42, "batch-size": 16, "image-size": 64, "local-epochs": 1}
    context = Context(run_id=1, node_id=1, node_config={"partition-id": 0, "num-partitions": 4},
                      state=RecordDict(), run_config=config)
    original = Net().state_dict()
    train_msg = Message(content=RecordDict({"arrays": ArrayRecord(original),
                        "config": ConfigRecord({"server-round": 1, "lr": 0.01})}),
                        dst_node_id=1, message_type="train")
    reply = app(train_msg, context)
    assert reply.content["metrics"]["num-examples"] == 50
    assert reply.content["metrics"]["train_loss"] > 0
    updated = reply.content["arrays"].to_torch_state_dict()
    assert any(not torch.equal(original[k], updated[k]) for k in original)
    eval_msg = Message(content=RecordDict({"arrays": ArrayRecord(updated),
                       "config": ConfigRecord({"server-round": 1})}),
                       dst_node_id=1, message_type="evaluate")
    result = app(eval_msg, context)
    assert result.content["metrics"]["num-examples"] == 10
    assert 0 <= result.content["metrics"]["eval_acc"] <= 1
    assert "arrays" not in result.content
