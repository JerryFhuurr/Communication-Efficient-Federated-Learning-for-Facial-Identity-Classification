# Federated compression experiments with Flower and PyTorch

This repository measures communication-efficient federated learning with a
replaceable supervised-classification task. It includes an aligned CelebA demo,
a generic class-folder image workflow, and a synthetic vector task for fast
end-to-end validation.

Implemented upload methods:

- uncompressed FedAvg
- QSGD
- QSGD followed by LLZ-p

LLZ-p with `p=0` has an exact replay gate: it must reconstruct the same QSGD
updates and training trajectory. Positive `p` values are evaluated as a separate
lossy method. LLZ-SI and Top-k are future research extensions.

See [FRAMEWORK.md](FRAMEWORK.md) for the task contract, experiment protocol,
recovery guarantees, reporting semantics, and known limitations.

## Project layout

| Path | Purpose |
| --- | --- |
| `compression/` | Standalone NumPy QSGD and LLZ codecs |
| `federated_compression/` | PyTorch-delta and Flower-record adapter |
| `flower_face/task.py` | Default image task: CNN, loader, training and evaluation |
| `flower_face/synthetic_task.py` | Small vector task used to validate reuse |
| `flower_face/client_app.py` | Flower client train/evaluate handlers |
| `flower_face/server_app.py` | FedAvg, validation, communication accounting and checkpoints |
| `flower_face/prepare_data.py` | CelebA subset manifest preparation |
| `flower_face/prepare_generic.py` | Class-folder image and synthetic manifest preparation |
| `flower_face/compression_study.py` | Resumable matched compression study |
| `flower_face/experiment_suite.py` | Resumable QSGD/LLZ parameter grid |
| `flower_face/report.py` | Audited JSON, CSV, Markdown and figure generation |
| `experiments/` | Versioned TOML experiment templates |
| `tests/` | Codec, Flower, reproducibility, dataset and workflow tests |

The historical `flower_face` package name remains to preserve Flower entry
points. Dataset and compression code are separated by the task interface in
`flower_face/task_api.py`.

## Setup

Python 3.11 is recommended. From the project root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,reports]"
```

Run all tests:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## CelebA demo

The local `img_align_celeba/`, `identity_CelebA.txt`, and
`list_eval_partition.txt` inputs are ignored by Git. Build the deterministic
subset manifest and run the default configuration:

```powershell
.venv\Scripts\python.exe -m flower_face.prepare_data --images img_align_celeba
.venv\Scripts\python.exe -m flower_face.run
```

The default run uses four simulated clients and writes an immutable experiment
directory under `outputs/`.

## Use another image dataset

Arrange images by class:

```text
C:\datasets\my-images\
├── class-a\
│   ├── 001.jpg
│   └── 002.jpg
└── class-b\
    ├── 001.jpg
    └── 002.jpg
```

Create deterministic IID client partitions:

```powershell
.venv\Scripts\python.exe -m flower_face.prepare_generic --images C:\datasets\my-images --output data\custom\manifest.json --clients 4
```

The preparer reports the number of classes. Set that value and the manifest path
in `experiments/images.toml`, then run the complete comparison:

```powershell
.venv\Scripts\python.exe -m flower_face.compression_study --config experiments\images.toml
```

No compression, Flower, aggregation, or reporting code needs to change.

## Validate the framework without images

```powershell
.venv\Scripts\python.exe -m flower_face.prepare_generic --synthetic --output data\synthetic\manifest.json
.venv\Scripts\python.exe -m flower_face.compression_study --config experiments\synthetic.toml
```

This runs the same Flower client/server and compression paths with a small MLP.

## Reports and parameter grids

Generate a report from a completed study:

```powershell
.venv\Scripts\python.exe -m flower_face.report outputs\compression-study-TIMESTAMP
```

Inspect a grid before starting it, then run it when its size is acceptable:

```powershell
.venv\Scripts\python.exe -m flower_face.experiment_suite --config experiments\synthetic-grid.toml --plan-only
.venv\Scripts\python.exe -m flower_face.experiment_suite --config experiments\synthetic-grid.toml
```

Resume interrupted work using its saved protocol:

```powershell
.venv\Scripts\python.exe -m flower_face.compression_study --resume outputs\compression-study-TIMESTAMP
.venv\Scripts\python.exe -m flower_face.experiment_suite --resume outputs\suite-TIMESTAMP
```

Communication results count serialized Flower objects, including codec and
metric metadata. They do not represent captured network traffic or runtime.
