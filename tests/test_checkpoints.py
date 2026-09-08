import json

import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict

from flower_face.experiment import Experiment
from flower_face import server_app


@pytest.fixture
def experiment_config(tmp_path):
    manifest = dict(image_root=str(tmp_path), image_variant="synthetic-test",
                    identities=[100, 200], num_clients=4, examples=[
                        dict(filename=f"{split}-{client}.jpg", label=0, identity=100,
                             split=split, client_id=client)
                        for split in ("train", "validation") for client in range(4)
                    ])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config = {"manifest": str(path), "num-classes": 2, "num-clients": 4, "seed": 42,
              "output-dir": str(tmp_path / "outputs"), "num-server-rounds": 2,
              "learning-rate": 0.01, "evaluate-final-test": False}
    return config, manifest


def test_selection_uses_loss_keeps_earliest_tie_and_preserves_weights(experiment_config):
    config, manifest = experiment_config
    record = Experiment(config["output-dir"], "test", config, manifest, "epoch")
    weights = {"weight": torch.tensor([1.0])}
    assert record.consider_best(weights, 1, 0.5, 0.6)
    weights["weight"].fill_(9)
    assert not record.consider_best(weights, 2, 0.5, 0.9)
    assert not record.consider_best(weights, 3, 0.6, 1.0)
    saved = torch.load(record.output / "best_model.pt", weights_only=True)
    assert saved["step"] == 1
    assert saved["state_dict"]["weight"].item() == 1.0
    with pytest.raises(ValueError, match="finite"):
        record.consider_best(weights, 4, float("nan"), 0.9)
    assert record.consider_best(weights, 5, 0.4, 0.4)
    saved = torch.load(record.output / "best_model.pt", weights_only=True)
    assert saved["step"] == 5
    assert saved["state_dict"]["weight"].item() == 9.0


class ScriptedGrid:
    """Exercise actual FedAvg dispatch/aggregation with controlled client results."""

    def get_node_ids(self):
        return [1, 2, 3, 4]

    def send_and_receive(self, messages, timeout):
        replies = []
        for msg in messages:
            step = msg.content["config"]["server-round"]
            if msg.metadata.message_type == "train":
                weights = msg.content["arrays"].to_torch_state_dict()
                content = RecordDict({
                    "arrays": ArrayRecord({k: torch.full_like(v, float(step)) for k, v in weights.items()}),
                    "metrics": MetricRecord({"train_loss": 1.0 / step, "num-examples": 50}),
                })
            else:
                # Later accuracy improves, but loss gets worse: select round 1.
                content = RecordDict({"metrics": MetricRecord({
                    "eval_loss": 0.5 if step == 1 else 0.75,
                    "eval_acc": 0.5 if step == 1 else 0.9, "num-examples": 10,
                })})
            content["client"] = ConfigRecord({"partition-id": msg.metadata.dst_node_id - 1})
            replies.append(Message(content=content, reply_to=msg))
        return replies


@pytest.mark.parametrize("evaluate_test", [False, True])
def test_full_strategy_saves_matching_best_and_final_and_tests_only_selected(
        experiment_config, monkeypatch, evaluate_test):
    config, _ = experiment_config
    config["evaluate-final-test"] = evaluate_test
    test_calls = []

    def load_test(config, split):
        assert evaluate_test and split == "test"
        return "test-loader"

    def evaluate_selected(model, loader):
        assert loader == "test-loader"
        assert all(torch.all(v == 1.0) for v in model.state_dict().values())
        test_calls.append(1)
        return 0.6, 0.5

    monkeypatch.setattr(server_app, "load_data", load_test)
    monkeypatch.setattr(server_app, "test", evaluate_selected)
    context = Context(run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config=config)
    server_app.main(ScriptedGrid(), context)
    from pathlib import Path
    output = next(Path(config["output-dir"]).iterdir())
    best = torch.load(output / "best_model.pt", weights_only=True)
    final = torch.load(output / "final_model.pt", weights_only=True)
    assert best["step"] == 1 and final["step"] == 2
    assert all(torch.all(v == 1.0) for v in best["state_dict"].values())
    assert all(torch.all(v == 2.0) for v in final["state_dict"].values())
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["best_checkpoint"]["step"] == 1
    assert len(test_calls) == int(evaluate_test)
    assert (metrics["test"] is not None) == evaluate_test
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert [r["new_best"] for r in history] == [True, False]
    assert json.loads((output / "experiment.json").read_text())["status"] == "completed"
    communication = json.loads((output / "communication.json").read_text())
    assert communication["measurement"]["network_traffic_measured"] is False
    assert communication["totals"] == metrics["communication"] == history[-1]["cumulative_communication"]
    messages = [json.loads(line) for line in (output / "communication_messages.jsonl").read_text().splitlines()]
    assert len(messages) == 32  # Four clients, two phases, two directions, two rounds.
    total = communication["totals"]["total"]
    assert total["messages"] == 32
    assert total["serialized_object_bytes"] == sum(row["serialized_object_bytes"] for row in messages)
    assert total["serialized_object_bits"] == 8 * total["serialized_object_bytes"]
    assert communication["totals"]["validation"]["uplink"]["raw_array_bytes"] == 0
    assert communication["totals"]["validation"]["uplink"]["serialized_object_bytes"] > 0
    for row in history:
        assert row["communication"] == communication["rounds"][str(row["round"])]
        for phase in ("train", "validation"):
            for direction in ("downlink", "uplink"):
                assert row["communication"][phase][direction]["messages"] == 4


def test_missing_clients_cannot_select_checkpoint(experiment_config):
    config, manifest = experiment_config
    record = Experiment(config["output-dir"], "test", config, manifest, "round")
    strategy = server_app.CheckpointFedAvg(record, 4)
    with pytest.raises(RuntimeError, match="all configured clients"):
        strategy.aggregate_train(1, [])
    assert record.best is None
