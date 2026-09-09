# Training diagnosis

These scripts reuse the existing CNN, preprocessing, SGD, and experiment records.
They do not modify the training packages or load held-out test images.

Run from the project root with the existing project environment:

```powershell
.venv\Scripts\python.exe scripts/diagnose_training.py
```

The default inputs are the completed uncompressed/QSGD seed study and the
three-seed lossless LLZ summary. Override their paths with `--study` and
`--llz-summary`. Each invocation creates a new output directory; `--output-dir`
must also name a directory that does not already exist.

The runner audits the six saved federated experiments, checks that `task.py` is
byte-identical to the study snapshot, and verifies that the earlier QSGD curves
exactly match the later LLZ comparison. It then starts three independent CPU
processes for centralized seeds 42, 43, and 44, with one PyTorch thread each.
Each seed must reproduce its federated initial-model hash. The test set remains
unused, and checkpoints are selected by minimum validation loss.

The training budget is 300 epochs/rounds and 60,000 example presentations per
method per seed. Centralized training has 3,900 sequential SGD steps; the
federated runs have 4,800 steps summed across clients, followed by 300 model
aggregations. Different shuffling, incomplete batches, and averaging mean these
are diagnostic comparisons at equal data exposure, not identical optimizer
trajectories. Concurrent centralized runs should not be used for wall-time
benchmarking against the earlier Flower runs.

Outputs include `diagnosis.json`, `curves.json`, and a directory per centralized
seed containing logs, metadata, every epoch's metrics/model hash, and best/final
checkpoints. The final audit verifies environment and initialization metadata,
checkpoint hashes, and selection. Existing federated checkpoints are also
evaluated on the training and validation sets for a consistent final-model view.
Historical federated optimization loss was measured during client training;
it is not the full training-set loss of the aggregated model.

To render unsmoothed PNG and SVG figures:

```powershell
uv pip install --python .venv/Scripts/python.exe --cache-dir .uv-cache -r scripts/requirements-diagnostics.txt
.venv\Scripts\python.exe scripts/plot_training_diagnosis.py outputs/training-diagnosis-<timestamp>
```

The plotter can also show the saved federated curves while centralized runs are
still active. On completion it produces `learning-curves.png/.svg` and
`centralized-generalization.png/.svg`. LLZ-p=0 shares QSGD's learning curve, so it
does not need another overlapping line. All checkpoint dots are selected by
validation loss, including when displayed on an accuracy plot.

## Controlled learning-rate sweep

```powershell
.venv\Scripts\python.exe scripts/sweep_learning_rate.py
```

This compares uncompressed FedAvg at rates 0.01, 0.03, and 0.1, using all three
reference seeds and 300 rounds. The completed 0.01 runs are re-audited and reused;
six new runs train sequentially. Each uses the original study's saved application
snapshot, so source, initialization, environment and training controls must match
its baseline apart from learning rate and output path. Training and test defaults
in the working project are not edited. `--rates` can specify a different
prospective candidate set but must include the reference rate.

The selection rule is recorded before training: lowest mean validation-selected
loss across seeds, with smaller learning rate winning exact ties. Accuracy is
reported at the same checkpoints. This is exploratory tuning on the existing
validation split, not a held-out estimate of the selected setting's quality.

Each new `outputs/lr-sweep-<timestamp>/` contains `sweep.json`, a frozen `app/`,
per-attempt logs and experiment artifacts, and final `summary.json` and
`curves.json`. Successful runs are re-evaluated on train/validation data and
audited for successful client participation, byte accounting and checkpoint
hashes. No test images are loaded. Failed attempts are retained separately;
an incomplete set cannot select a winning learning rate.

To resume a failed/interrupted sweep without repeating completed runs:

```powershell
.venv\Scripts\python.exe scripts/sweep_learning_rate.py --resume outputs/lr-sweep-<timestamp>
```

Resume uses the saved protocol and rejects a changed runner, reference report,
or application snapshot. It re-audits completed runs before continuing. To
generate the completed sweep's raw learning curves and aggregate comparison:

```powershell
.venv\Scripts\python.exe scripts/plot_lr_sweep.py outputs/lr-sweep-<timestamp>
```

This writes `learning-rate-curves.png/.svg` and `learning-rate-summary.png/.svg`.
Summary error bars show sample standard deviations across seeds, not confidence
intervals. Plotting uses the same optional Matplotlib dependency listed above.

## Controlled weight-decay sweep

```powershell
.venv\Scripts\python.exe scripts/sweep_weight_decay.py
```

This runs uncompressed FedAvg at learning rate 0.1, comparing SGD weight decay
0, 0.0001 and 0.001 for seeds 42, 43 and 44. All nine runs use 300 rounds and
the same new application snapshot. Before any positive-decay experiment, the
three zero-decay runs must reproduce every model hash, learning metric and
selected checkpoint from the saved rate-0.1 study. This validates the changed
training code against the earlier implementation over all 900 baseline rounds.

The regularizer is PyTorch SGD's coupled L2 weight decay, applied to all model
parameters including biases. The recorded loss remains cross-entropy; validation
loss does not include the parameter penalty. The project defaults remain learning
rate 0.01 and weight decay 0.0. Missing weight-decay configuration means zero,
preserving callers of the earlier training function.

The selection rule is declared before training: lowest mean validation-selected
loss across seeds, with smaller decay breaking exact ties. `--decays` can set a
different prospective candidate list, which must include zero. `--reference`
can specify the completed learning-rate sweep containing the legacy 0.1 runs.

Each `outputs/wd-sweep-<timestamp>/` records the protocol, old/new source hashes,
zero-decay replay results, complete experiments, train/validation checkpoint
evaluations, and raw plotting data. To resume after an interruption or generate
the completed plots without retraining:

```powershell
.venv\Scripts\python.exe scripts/sweep_weight_decay.py --resume outputs/wd-sweep-<timestamp>
.venv\Scripts\python.exe scripts/plot_lr_sweep.py outputs/wd-sweep-<timestamp> --parameter weight-decay
```

The generalized plotter writes `weight-decay-curves.png/.svg` and
`weight-decay-summary.png/.svg`; its default remains learning-rate plots for
older sweep outputs. Individual runs also accept these overrides:

```powershell
.venv\Scripts\python.exe -m flower_face.run --rounds 300 --seed 42 --compression none --lr 0.1 --weight-decay 0.001 --skip-test
```

`flower_face.check_training` also accepts `--weight-decay` for centralized or
tiny-batch checks. Historical diagnostic runners that demand byte-identical
`task.py` will reject the new training source against their older snapshots;
their saved plots remain reproducible without rerunning training. Use the
weight-decay runner's explicit zero-decay replay checks to compare this source
revision with the earlier baseline.

## Controlled training-only augmentation sweep

```powershell
.venv\Scripts\python.exe scripts/sweep_augmentation.py
```

This compares no augmentation with `RandomHorizontalFlip(p=0.5)` at learning
rate 0.1 and weight decay 0 for seeds 42, 43 and 44. Flipping is applied after
resize only while reading training images. Validation, test, and full-dataset
checkpoint evaluation use the unchanged deterministic transform.

Before any augmented run, the runner executes three no-augmentation runs and
requires exact equality with the saved weight-decay sweep: all 300 model hashes,
training/validation metrics, and selected checkpoint per seed. This checks that
adding the transform option preserves disabled behavior over 900 rounds.

The predeclared selection rule chooses the lowest mean validation-selected loss
across seeds; no augmentation wins an exact tie. The held-out test remains
unused. To resume a stopped sweep or regenerate completed PNG/SVG figures:

```powershell
.venv\Scripts\python.exe scripts/sweep_augmentation.py --resume outputs/augmentation-sweep-<timestamp>
.venv\Scripts\python.exe scripts/plot_augmentation_sweep.py outputs/augmentation-sweep-<timestamp>
```

Outputs include the protocol and attempts in `sweep.json`, aggregate selection
in `summary.json`, raw curves and checkpoint evaluations in `curves.json`, a
frozen application, and one directory per run. The plotter writes
`augmentation-curves.png/.svg` and `augmentation-summary.png/.svg`.

Individual runs accept `--augmentation none` or `--augmentation
horizontal-flip`. After the completed sweep selected horizontal flipping, the
project defaults were frozen at learning rate 0.1, weight decay 0, and horizontal
flipping for the next thesis stage. Historical experiments continue to use
their saved snapshots and configurations.
