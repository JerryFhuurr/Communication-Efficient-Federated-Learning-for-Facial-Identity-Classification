"""Experiment records and validation-based checkpoints, independent of Flower."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time

import torch
from flower_face.reproducibility import settings, source_hash


def atomic_replace(temporary, target):
    """Allow brief Windows reader/scanner locks; still fail on persistent errors."""
    for attempt in range(8):
        try:
            temporary.replace(target)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                raise
            time.sleep(min(0.01 * 2**attempt, 0.2))


class Experiment:
    def __init__(self, output_root, mode, config, manifest, step_kind):
        self.started = time.perf_counter()
        self.config = dict(config)
        self.manifest = manifest
        self.step_kind = step_kind
        self.best = None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.output = Path(output_root) / f"{mode}-{stamp}"
        self.output.mkdir(parents=True, exist_ok=False)
        self.manifest_hash = hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest()
        self.metadata = dict(
            mode=mode, status="incomplete", started_at_utc=stamp, config=self.config,
            step_kind=step_kind, selection_rule="lowest validation loss; earliest step wins ties",
            manifest_sha256=self.manifest_hash, image_variant=manifest["image_variant"],
            split_counts=dict(Counter(row["split"] for row in manifest["examples"])),
            versions={name: version(name) for name in ("flwr", "torch", "torchvision", "numpy", "ray")},
            python=platform.python_version(), platform=platform.platform(),
            reproducibility=settings(), source_sha256=source_hash(),
        )
        self.write_json("experiment.json", self.metadata)
        self.write_json("manifest.json", manifest)
        print(f"Experiment directory: {self.output.resolve()}", flush=True)

    def write_json(self, name, data):
        target = self.output / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        atomic_replace(temporary, target)

    def save_checkpoint(self, name, state_dict, step, validation=None):
        target = self.output / name
        temporary = target.with_suffix(".pt.tmp")
        torch.save(dict(
            state_dict={k: v.detach().cpu().clone() for k, v in state_dict.items()},
            config=self.config, identities=self.manifest["identities"],
            step=step, step_kind=self.step_kind, validation=validation,
            manifest_sha256=self.manifest_hash,
        ), temporary)
        atomic_replace(temporary, target)

    def consider_best(self, state_dict, step, loss, accuracy):
        if not math.isfinite(loss) or not math.isfinite(accuracy):
            raise ValueError("Validation metrics must be finite; checkpoint selection stopped.")
        if self.best is not None and loss >= self.best["validation_loss"]:
            return False
        self.best = dict(step=step, step_kind=self.step_kind, validation_loss=loss,
                         validation_accuracy=accuracy, checkpoint="best_model.pt")
        self.save_checkpoint("best_model.pt", state_dict, step, self.best)
        self.write_json("best_checkpoint.json", self.best)
        print(f"Best checkpoint: {self.step_kind} {step}, validation loss={loss:.4f}, "
              f"accuracy={accuracy:.1%}", flush=True)
        return True

    def log_step(self, row):
        row = dict(row, elapsed_seconds=time.perf_counter() - self.started)
        with (self.output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def complete(self, metrics):
        self.write_json("metrics.json", dict(metrics, best_checkpoint=self.best))
        self.metadata.update(status="completed", elapsed_seconds=time.perf_counter() - self.started,
                             best_checkpoint=self.best)
        self.write_json("experiment.json", self.metadata)
