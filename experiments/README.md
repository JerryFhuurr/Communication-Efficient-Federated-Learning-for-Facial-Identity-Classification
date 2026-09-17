# Experiment configurations

These TOML files separate experiment choices from Python implementation code.

`digiface-head-confirmed.toml` fixes the selected linear-head settings for seeds
42/43/44 after uncompressed validation confirmation (72.47% mean, 0.46 percentage
points sample SD). Running this template starts a compression comparison; the
confirmation itself evaluated only the uncompressed model. Test data is unused.

`digiface-head-selected.toml` records the provisional winner of the subsequent
three-trial head-improvement batch (120 rounds, higher LR, or nonlinear head).
Earlier selected configurations remain available for reproducibility.
`flower_face.pretrained_mlp_task` uses the same feature cache with a
512-to-256-to-class ReLU head; its larger update size must be reported separately.

- `images.toml` is the starting point for any class-folder image dataset.
- `digiface.toml` runs the prepared 100-identity DigiFace-1M subset.
- `digiface-tuned.toml` records the provisional 60-round, horizontal-flip settings
  selected using one uncompressed seed-42 pilot (38.0% validation accuracy).
  Running it starts a four-method comparison for seed 42; that comparison has
  not yet been run. The original `digiface.toml` preserves the 30-round study.
- `digiface-pretrained.toml` uses cached ImageNet ResNet-18 features and a trainable
  linear head. It requires the feature preparation described in the main README.
  Backbone weights and extraction costs are outside federated message accounting.
- `digiface-pretrained-selected.toml` records the provisional winner of the
  single-seed head-tuning batch (training duration, learning rate, weight decay).
  It is a future compression-comparison configuration, not evidence that those
  compressed runs have already been performed. The original pretrained template
  preserves the first 57.3% pilot settings.
- `synthetic.toml` is a fast end-to-end check of every compression path.
- `synthetic-grid.toml` demonstrates a QSGD-level and LLZ-window grid.

Copy a file before changing it for a thesis experiment so the committed template
remains stable. Paths are resolved from the project root. Run a single study with:

```powershell
.venv\Scripts\python.exe -m flower_face.compression_study --config experiments/images.toml
```

Inspect a grid before starting it:

```powershell
.venv\Scripts\python.exe -m flower_face.experiment_suite --config experiments/synthetic-grid.toml --plan-only
```
