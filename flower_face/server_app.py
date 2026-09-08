"""Ordinary full-participation FedAvg and a final held-out test evaluation."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import torch

from flower_face.task import Net, load_data, read_manifest, seed_everything, test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):
    config = dict(context.run_config)
    manifest = read_manifest(config["manifest"], config["num-classes"], config["num-clients"])
    seed_everything(config["seed"])
    model = Net(config["num-classes"])
    clients = config["num-clients"]
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0,
                      min_train_nodes=clients, min_evaluate_nodes=clients,
                      min_available_nodes=clients)
    result = strategy.start(
        grid=grid, initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({"lr": config["learning-rate"]}),
        num_rounds=config["num-server-rounds"], timeout=120,
    )
    expected_rounds = set(range(1, config["num-server-rounds"] + 1))
    if (set(result.train_metrics_clientapp) != expected_rounds
            or set(result.evaluate_metrics_clientapp) != expected_rounds):
        raise RuntimeError("Some rounds have no train/evaluation results; inspect Flower logs.")
    model.load_state_dict(result.arrays.to_torch_state_dict())
    # Test images are evaluated only once after all training/validation rounds.
    test_metrics = None
    if config.get("evaluate-final-test", True):
        loss, accuracy = test(model, load_data(config, split="test"))
        test_metrics = {"loss": loss, "accuracy": accuracy}
    output = Path(config["output-dir"]) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": model.state_dict(), "identities": manifest["identities"],
                "config": config}, output / "final_model.pt")
    metrics = dict(config=config, image_variant=manifest["image_variant"],
                   versions={name: version(name) for name in ("flwr", "torch", "torchvision", "numpy", "ray")},
                   manifest_sha256=hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest(),
                   identities=manifest["identities"],
                   train={str(k): dict(v) for k, v in result.train_metrics_clientapp.items()},
                   validation={str(k): dict(v) for k, v in result.evaluate_metrics_clientapp.items()},
                   test=test_metrics,
                   model_parameters=sum(p.numel() for p in model.parameters()))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if test_metrics is not None:
        print(f"Final held-out test: loss={test_metrics['loss']:.4f}, accuracy={test_metrics['accuracy']:.2%}")
    else:
        print("Held-out test skipped; use validation metrics for development.")
    print(f"Saved checkpoint, metrics, and manifest to {output.resolve()}")
