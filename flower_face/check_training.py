"""Overfit a tiny training batch or train centrally; never read the test split."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tomllib

import torch
from torch.utils.data import DataLoader, Subset

from flower_face.task import Net, load_data, read_manifest, seed_everything, test, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["overfit", "centralized"])
    parser.add_argument("--steps", type=int, default=500, help="Maximum tiny-batch updates")
    parser.add_argument("--epochs", type=int, default=30, help="Centralized training epochs")
    parser.add_argument("--lr", type=float, help="Default: 0.1 for overfit; project LR for centralized")
    args = parser.parse_args()
    if args.steps < 1 or args.epochs < 1 or (args.lr is not None and args.lr <= 0):
        parser.error("Steps, epochs, and learning rate must be positive")
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    config["manifest"] = str((root / config["manifest"]).resolve())
    manifest = read_manifest(config["manifest"], config["num-classes"], config["num-clients"])
    seed_everything(config["seed"])
    model = Net(config["num-classes"])
    loader = load_data(config, split="train")
    lr = args.lr if args.lr is not None else (0.1 if args.mode == "overfit" else config["learning-rate"])
    if args.mode == "overfit":
        # One fixed training image per identity; no validation/test samples.
        seen, indices = set(), []
        for index, row in enumerate(loader.dataset.rows):
            if row["label"] not in seen:
                seen.add(row["label"])
                indices.append(index)
        if len(seen) != config["num-classes"]:
            raise ValueError("Training split must include every identity")
        dataset = Subset(loader.dataset, indices)
        loader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
        evaluation_loader = loader
        validation_loader = None
        iterations = args.steps
        selected = [loader.dataset.dataset.rows[i] for i in indices]
    else:
        evaluation_loader = DataLoader(loader.dataset, batch_size=config["batch-size"], shuffle=False)
        validation_loader = load_data(config, split="validation")
        iterations = args.epochs
        selected = loader.dataset.rows
    print(f"{args.mode}: {len(loader.dataset)} training images, SGD lr={lr}, CPU", flush=True)
    history = []
    for iteration in range(1, iterations + 1):
        # task.train uses stateless SGD (no momentum), so one-epoch calls preserve
        # the same optimizer behavior as a single multi-epoch call.
        optimization_loss = train(model, loader, epochs=1, lr=lr)
        train_loss, train_acc = test(model, evaluation_loader)
        row = dict(iteration=iteration, optimization_loss=optimization_loss,
                   train_loss=train_loss, train_accuracy=train_acc)
        if validation_loader is not None:
            val_loss, val_acc = test(model, validation_loader)
            row.update(validation_loss=val_loss, validation_accuracy=val_acc)
        history.append(row)
        if args.mode == "centralized" or iteration == 1 or iteration % 25 == 0 or iteration == iterations:
            message = f"{iteration:4d}/{iterations}: train loss={train_loss:.4f}, accuracy={train_acc:.1%}"
            if validation_loader is not None:
                message += f" | validation loss={val_loss:.4f}, accuracy={val_acc:.1%}"
            print(message, flush=True)
        if args.mode == "overfit" and train_acc == 1.0 and train_loss < 0.05:
            print(f"Tiny-batch check reached 100% accuracy and loss < 0.05 at step {iteration}.", flush=True)
            break
    if args.mode == "overfit" and not (history[-1]["train_accuracy"] == 1.0 and history[-1]["train_loss"] < 0.05):
        print("Tiny batch has not reached the target yet; inspect the learning curve before proceeding.", flush=True)
    output = root / "outputs" / f"{args.mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    output.mkdir(parents=True)
    metrics = dict(mode=args.mode, config=config, learning_rate=lr, history=history,
                   image_variant=manifest["image_variant"], training_examples=selected,
                   manifest_sha256=hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest(),
                   test_evaluated=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save(dict(state_dict=model.state_dict(), config=config, identities=manifest["identities"]),
               output / "final_model.pt")
    print(f"Saved results: {output}", flush=True)


if __name__ == "__main__":
    main()
