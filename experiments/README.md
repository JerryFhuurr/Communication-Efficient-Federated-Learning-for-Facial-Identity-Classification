# Experiment configurations

These TOML files separate experiment choices from Python implementation code.

- `images.toml` is the starting point for any class-folder image dataset.
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
