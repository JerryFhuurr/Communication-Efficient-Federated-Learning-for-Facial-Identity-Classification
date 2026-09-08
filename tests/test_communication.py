import json
from types import SimpleNamespace

import numpy as np
import pytest
from flwr.app import ArrayRecord, ConfigRecord, Error, Message, MetricRecord, RecordDict
from flwr.supercore.inflatable.inflatable_object import get_all_nested_objects
from flwr.supercore.inflatable.inflatable_utils import inflate_object_from_contents

from flower_face.communication import CommunicationLedger, MeasuredGrid, measure_message


def instruction():
    return Message(RecordDict({
        "arrays": ArrayRecord([np.arange(6, dtype=np.float32).reshape(2, 3),
                               np.array(17, dtype=np.int64)]),
        "config": ConfigRecord({"server-round": 1, "lr": 0.01}),
    }), dst_node_id=7, message_type="train")


def test_measured_bytes_reconstruct_complete_message_with_mixed_dtypes():
    message = instruction()
    # Use Flower's own transport traversal and inflation as an independent oracle.
    encoded = {key: obj.deflate() for key, obj in get_all_nested_objects(message).items()}
    sizes = measure_message(message)
    assert sizes["raw_array_bytes"] == 6 * 4 + 8
    assert sizes["array_data_bytes"] > sizes["raw_array_bytes"]  # NumPy headers
    assert sizes["serialized_object_bytes"] == sum(map(len, encoded.values()))
    assert sizes["serialized_object_bits"] == 8 * sum(map(len, encoded.values()))
    assert sum(v["bytes"] for v in sizes["objects_by_type"].values()) == sum(map(len, encoded.values()))
    restored = inflate_object_from_contents(message.object_id, dict(encoded))
    assert restored.content["config"] == message.content["config"]
    for key, array in message.content["arrays"].items():
        np.testing.assert_array_equal(restored.content["arrays"][key].numpy(), array.numpy())
    assert encoded == {key: obj.deflate() for key, obj in get_all_nested_objects(message).items()}


def test_identical_arrays_deduplicate_only_inside_serialized_graph():
    values = np.ones(1000, dtype=np.float32)
    message = Message(RecordDict({"arrays": ArrayRecord([values, values.copy()])}),
                      dst_node_id=7, message_type="train")
    sizes = measure_message(message)
    assert sizes["raw_array_bytes"] == 8000
    assert sizes["objects_by_type"]["Array"]["objects"] == 1
    assert sizes["objects_by_type"]["ArrayChunk"]["objects"] == 1
    assert sizes["serialized_object_bytes"] < sizes["raw_array_bytes"]


def test_metrics_and_error_replies_have_nonzero_serialized_cost():
    message = instruction()
    reply = Message(RecordDict({"metrics": MetricRecord({"eval_loss": 0.5,
                                                       "eval_acc": 0.4, "num-examples": 10})}),
                    reply_to=message)
    before = measure_message(reply)
    assert before["raw_array_bytes"] == before["array_data_bytes"] == 0
    assert before["serialized_object_bytes"] > 0
    reply.content["metrics"]["additional-metadata"] = [1.0] * 100
    assert measure_message(reply)["serialized_object_bytes"] > before["serialized_object_bytes"]
    error = Message(error=Error(code=1, reason="client failed"), reply_to=message)
    sizes = measure_message(error)
    assert sizes["raw_array_bytes"] == 0
    assert sizes["objects_by_type"].keys() == {"Message"}
    assert sizes["serialized_object_bytes"] > 0


def make_ledger(tmp_path):
    def write_json(name, data):
        (tmp_path / name).write_text(json.dumps(data))
    return CommunicationLedger(SimpleNamespace(output=tmp_path, metadata={}, write_json=write_json))


def test_each_recipient_and_round_is_counted_even_with_identical_content(tmp_path):
    ledger = make_ledger(tmp_path)
    message = instruction()
    size = measure_message(message)["serialized_object_bytes"]
    ledger.observe([message] * 4, 1, "train", "downlink")
    ledger.observe([message] * 4, 2, "train", "downlink")
    assert ledger.summary(1)["total"]["serialized_object_bytes"] == 4 * size
    assert ledger.summary()["total"]["serialized_object_bytes"] == 8 * size
    assert ledger.summary()["uplink"]["messages"] == 0


def test_grid_failure_preserves_submitted_messages_and_partial_error_reply(tmp_path):
    ledger = make_ledger(tmp_path)

    class FailingGrid:
        def send_and_receive(self, messages, timeout):
            # Emulate routing metadata assigned by Flower inside dispatch.
            messages[0].metadata.__dict__["_run_id"] = 1234
            messages[0].metadata.__dict__["_message_id"] = messages[0].object_id
            yield Message(error=Error(code=1, reason="failed"), reply_to=messages[0])
            raise RuntimeError("connection interrupted")

    grid = MeasuredGrid(FailingGrid(), ledger)
    with pytest.raises(RuntimeError, match="connection interrupted"):
        grid.send_and_receive([instruction()], timeout=1)
    rows = [json.loads(line) for line in (tmp_path / "communication_messages.jsonl").read_text().splitlines()]
    assert [row["direction"] for row in rows] == ["downlink", "uplink"]
    assert rows[0]["run_id"] == 1234 and rows[0]["message_id"]
    assert ledger.summary()["uplink"]["error_messages"] == 1
    assert ledger.summary()["total"]["messages"] == 2
    assert json.loads((tmp_path / "communication.json").read_text())["totals"] == ledger.summary()
