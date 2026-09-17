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

For the local DigiFace-1M partition, prepare the versioned experiment input
without copying its images:

```powershell
.venv\Scripts\python.exe -m flower_face.prepare_generic --images data\subjects_0-1999_72_imgs --output data\digiface-100\manifest.json --clients 4 --max-classes 100 --max-images-per-class 50
.venv\Scripts\python.exe -m flower_face.compression_study --config experiments\digiface.toml
```

Numeric identity folders use natural ordering, so this selects subjects 0–99.
The 5,000-image manifest stores deterministic splits and source-file hashes.

## Frozen pretrained image model

`flower_face.pretrained_task` trains a linear 512-to-class head on cached
ResNet-18 features. The backbone uses TorchVision's fixed
`ResNet18_Weights.IMAGENET1K_V1` ImageNet weights, **not face-specific weights**.
Preprocessing follows the weights' resize/crop-to-224 and ImageNet normalization.
The manifest retains every original identity, split and client assignment.
Original and horizontally flipped features are cached; only training samples
randomly choose a view. Validation/test always use the original view.

```powershell
.venv\Scripts\python.exe -m flower_face.prepare_pretrained --manifest data\digiface-100\manifest.json --output data\digiface-resnet18
.venv\Scripts\python.exe -m flower_face.run --task-module flower_face.pretrained_task --manifest data\digiface-resnet18\manifest.json --num-classes 100 --num-clients 4 --image-size 224 --rounds 60 --seed 42 --lr 0.01 --augmentation horizontal-flip --compression none --skip-test --output-dir outputs\digiface-pretrained-pilot
```

Preparation can use `--weights PATH` for a downloaded official weights file;
its checksum is verified. It refuses to overwrite a nonempty output directory.
Keep the source manifest and prepared directory, including the weights, to
reproduce inference. A head checkpoint alone is not an end-to-end face model.

At 100 classes, 51,300 trainable parameters (205,200 raw FP32 bytes) are
federated. The frozen backbone is not transmitted each round. This is a
head-only transfer-learning experiment, not full-backbone fine-tuning; do not
attribute reduced message sizes to compression. Weight distribution and feature
extraction costs are recorded in the prepared manifest and excluded from Flower
message totals. Preparation uses fixed weights only; no statistics or parameters
are fit to validation/test samples. The preprocessing cache is an offline local
optimization, not a private/distributed feature-extraction implementation.

Reference: [TorchVision ResNet-18 weights](https://docs.pytorch.org/vision/0.13/models/generated/torchvision.models.resnet18.html).

For inference on a new, already cropped face, use the selected head checkpoint
and the same prepared manifest. No feature cache is needed for inference, but the
manifest and its local backbone weight file must be kept together:

```powershell
$selectedHead = (Get-Content outputs\pretrained-tuning-batch\comparison.json -Raw | ConvertFrom-Json).selected_checkpoint
.venv\Scripts\python.exe -m flower_face.predict_pretrained --checkpoint "$selectedHead" --manifest data\digiface-resnet18\manifest.json --image path\to\cropped-face.png --top-k 5
```

Replace the image path with your input. Predictions are restricted to the
manifest's known identities; this command neither detects faces nor rejects
unknown people. Softmax scores are not calibrated confidence estimates.
The loader verifies the checkpoint/manifest association and frozen-weight hash
before reconstructing the complete image-to-label pipeline.

The predictor also supports `flower_face.pretrained_mlp_task`, a 512-to-256-to-class
ReLU head on the same cached features. It selects the architecture from the saved
checkpoint. For the newer head-improvement batch, obtain `selected_checkpoint`
from `outputs/head-improvement-batch/comparison.json` instead of the earlier batch.
At 100 classes the nonlinear head has 157,028 trainable parameters, compared with
51,300 for the linear head; the pretrained backbone remains frozen in both cases.

## Fine-tune the final ResNet stage

`flower_face.finetune_task` trains ResNet-18 `layer4` and the linear classifier.
The earlier layers stay frozen, so `prepare_finetune` caches their spatial output
once (float32, original and horizontal-flip views). The 5,000-image demo cache
uses about 2 GB. Existing splits and identity labels are preserved; extraction
does not fit on the data or evaluate the test split.

```powershell
$warmHead = (Get-Content outputs\head-improvement-batch\comparison.json -Raw | ConvertFrom-Json).selected_checkpoint
.venv\Scripts\python.exe -m flower_face.prepare_finetune --source-manifest data\digiface-resnet18\manifest.json --output data\digiface-resnet18-layer3 --head-checkpoint "$warmHead"
.venv\Scripts\python.exe -m flower_face.finetune_study --manifest data\digiface-resnet18-layer3\manifest.json --baseline (Split-Path "$warmHead") --output outputs\finetune-layer4-batch
```

Reuse a completed intermediate cache; preparation requires an empty destination.
The bounded batch runs 15 additional rounds at layer4 learning rates 0.001 and
0.003 on seed 42. The classifier uses 10 times that rate. If the best validation
loss improves over the warm-start baseline, the selected setting runs on seeds
43 and 44. All runs use the same selected seed-42 head, so confirmation varies
fine-tuning randomness only. The runner saves provenance, audited results and
plots under its output directory and refuses to overwrite it.

BatchNorm affine parameters train, but running statistics remain fixed. The
current task accepts uncompressed training only: compression of its integer
BatchNorm buffers needs an explicit policy before integration. CPU clients have
a 900-second response timeout. Only layer4/head state is federated; prefix
distribution, cache extraction and earlier warm-start training are excluded
from the additional rounds' communication counts.

If a batch stops, repeat its command with `--resume`. Completed results are
verified and reused; interrupted attempts remain intact, and only incomplete
settings restart in separate attempt directories. The CLI prevents automatic
Windows idle sleep while running and restores the request on exit; it does not
prevent manually putting the computer to sleep.

For another dataset, first prepare its generic manifest and frozen ResNet feature
cache, then pass that feature manifest here. A warm-start head must match its
identity mapping and manifest exactly; omit `--head-checkpoint` to prepare a cache
for standalone training with a random head. `finetune_study` specifically requires
a matching seed-42 warm-start baseline. A standalone run is also available:

```powershell
.venv\Scripts\python.exe -m flower_face.run --task-module flower_face.finetune_task --manifest data\digiface-resnet18-layer3\manifest.json --num-classes 100 --num-clients 4 --image-size 224 --rounds 15 --lr 0.001 --compression none --skip-test --output-dir outputs\finetune-single
```

Use `flower_face.predict_pretrained` with the fine-tuned checkpoint and intermediate
manifest to predict directly from a cropped image. It reconstructs the frozen
prefix and trained final stage; the intermediate feature cache is not needed
for prediction, but the manifest and verified original weights are required.

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
