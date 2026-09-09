# Reusable QSGD + LLZ framework

CelebA is an example workload. No further dataset tuning is required for the
framework. The supported task is supervised classification with sample-weighted
cross-entropy loss and accuracy. All clients participate each round.

## Boundaries

* `compression/`: NumPy codecs and a packet-backend registry. `PacketCodec`
  provides `pack`, `unpack`, `stats`, a wire identifier, and whether encoding
  introduces secondary distortion. The built-ins encode QSGD integer tensors.
* `federated_compression/`: named PyTorch model deltas and validated Flower
  records. Explicit random streams keep codec randomness separate from training.
* `flower_face/task_api.py`: module-based task contract. The runtime selects
  `task-module`; its code is bundled and hashed with the application.
* `flower_face/`: Flower handlers, FedAvg, checkpoints, studies and reports.
  The package name remains for compatibility with existing commands.

The supported compression path requires floating float32/float64 state tensors.
Models with BatchNorm integer counters require an explicit buffer aggregation
policy before use. Models without integer buffers, such as the provided CNN
and MLP, work directly. Regression and multi-label tasks require adapting the
loss/metric and checkpoint-selection contract. Top-k and LLZ-SI are not implemented.
The packet registry is for QSGD code encoders; a different quantizer requires an
update policy as well. No paper-specific algorithm has been guessed.

## Another image dataset

Organize files as `images/<class-name>/<image>.jpg` (PNG, JPEG, BMP also supported).
The preparer reads every image, rejects duplicate decoded RGB content, assigns
labels, makes disjoint train/validation/test splits, and partitions train and
validation images round-robin by class across clients. This is an IID starter
partition; subject/session leakage must be addressed with a domain-specific
manifest when multiple files share the same subject or acquisition session.

```powershell
.venv\Scripts\python.exe -m flower_face.prepare_generic --images C:\datasets\my-images --output data/custom/manifest.json --clients 4
```

Edit `experiments/images.toml`: set `num-classes` to the number printed by the
preparer and `manifest` to the new file. The image root is absolute. File hashes
are recorded and checked while loading, preventing changed images from silently
entering an experiment. Existing manifests are never overwritten by preparation.

```powershell
.venv\Scripts\python.exe -m flower_face.compression_study --config experiments/images.toml
```

No codec edits are needed. RGB conversion, resizing, normalization and the CNN
are supplied by the default task. Augmentation is explicitly disabled in this
template; enable it only when appropriate for the new dataset.

## Another model or input type

Add a Python module under `flower_face/` and set `task-module` in the `[task]`
table. Implement these callables, following `synthetic_task.py`:

* `Net(num_classes)` returns a PyTorch model with named state tensors.
* `read_manifest(path, num_classes, num_clients)` validates the dataset and
  returns metadata: `identities` (ordered class names), `image_variant` (task
  description), `examples` (split records), and `num_clients`.
* `load_data(config, client_id=None, split='train', seed=None)` returns a loader
  with a sized dataset. A missing client ID means the complete split.
* `train(model, loader, epochs, lr, *, weight_decay=0.0)` returns mean local loss.
* `test(model, loader)` returns `(mean_loss, accuracy)`.

The historical `identities` and `image_variant` field names are metadata, not a
requirement for face images. The synthetic example uses ordinary vector classes.
Use the supplied explicit seed for shuffle/augmentation reproducibility.
Declare any new task configuration keys in `[tool.flwr.app.config]` in
`pyproject.toml` before overriding them: Flower rejects undeclared run keys.

```powershell
.venv\Scripts\python.exe -m flower_face.prepare_generic --synthetic --output data/synthetic/manifest.json
.venv\Scripts\python.exe -m flower_face.compression_study --config experiments/synthetic.toml
```

## Experiment specifications and recovery

TOML `[study]` declares seeds, rounds, QSGD levels, LLZ tolerances and window.
`[task]` overrides the application defaults. Paths are relative to the project
root. CLI study options override TOML values. Use separate specification files
for different level/window combinations, preserving an uncompressed and QSGD
control for each comparison. This avoids confounding codec and training changes.

```powershell
.venv\Scripts\python.exe -m flower_face.compression_study --config experiments/synthetic.toml --qsgd-levels 127 --llz-window 128
.venv\Scripts\python.exe -m flower_face.compression_study --resume outputs/compression-study-TIMESTAMP
```

Resume uses the saved protocol and checks completed artifacts and the frozen
source. It skips completed runs and restarts an unfinished run from round one
in a new attempt directory; it does not restore optimizer/RNG state mid-run.
Retain the original runner version when resuming. Lossless replay is checked
again when loading completed pairs. Positive-p runs follow their own trajectory;
their later QSGD draws are applied to different model deltas.

## Reporting

For multiple levels/windows in one command, add a `[grid]` table as in
`experiments/synthetic-grid.toml`. Every cell is a complete matched study;
controls are deliberately repeated so each cell has its own source/config audit.

```powershell
.venv\Scripts\python.exe -m flower_face.experiment_suite --config experiments/synthetic-grid.toml --plan-only
.venv\Scripts\python.exe -m flower_face.experiment_suite --config experiments/synthetic-grid.toml
.venv\Scripts\python.exe -m flower_face.experiment_suite --resume outputs/suite-TIMESTAMP
```

Use the report command below on each completed cell's study directory. No grid
is run automatically: inspect its size with `--plan-only` before allocating time.

Install plotting dependencies once using `pip install -e ".[dev,reports]"` in
the project environment. Generate a report without training:

```powershell
.venv\Scripts\python.exe -m flower_face.report outputs/compression-study-TIMESTAMP
```

The `report/` directory contains Markdown, JSON, per-seed/per-round CSVs, and
PNG/SVG plots for accuracy, loss, accuracy versus serialized bits, communication,
and separate QSGD/LLZ distortion. The reporter first reconciles checkpoints,
history, configuration and communication using the study audit.

Bytes are logical serialized Flower objects, including packet and metric
metadata, not captured TCP/IP traffic. Distortion ratios use different reference
norms and cannot be added. Plot shading is sample SD across seeds, not a
confidence interval. This report does not establish statistical equivalence.

## Verification and extension

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Tests include codec known answers, malformed packets, lossless replay, actual
Flower handlers and weighted aggregation on a vector task, and class-folder
preparation/loading. The default CPU implementation and pinned Flower/PyTorch
versions define the supported runtime. Native Windows Ray can emit shutdown
exceptions; distinguish these from failed client replies and incomplete runs.
No inference of transport delivery or speedup is made from smaller packets.
