# Flower + PyTorch CelebA baseline

A small, local **closed-set identity classifier**: identify which of ten selected
CelebA identities is pictured. This is not face verification or recognition of
previously unseen people. The CNN starts from random weights.

The structure follows the [current Flower PyTorch tutorial](https://flower.ai/docs/framework/tutorial-series-write-your-first-flower-app-pytorch.html),
using `ClientApp`, `ServerApp`, the Message API, and built-in `FedAvg`.
Flower is pinned to 1.36.0; this project runs locally with four simulated clients.

## Files

| File | Responsibility |
| --- | --- |
| `flower_face/task.py` | Small CNN, image dataset, PyTorch training and evaluation; no Flower imports |
| `flower_face/client_app.py` | Receive weights, train on one client, return weights and sample-weighted metrics |
| `flower_face/server_app.py` | Initialize weights, run ordinary FedAvg, evaluate final test set, save results |
| `flower_face/prepare_data.py` | Reproducible identity selection and disjoint image splits |
| `flower_face/run.py` | Thin wrapper around `flwr run . local --stream`; supplies absolute local paths and four clients |
| `pyproject.toml` | Dependencies, Flower entry points, experiment defaults |
| `tests/test_baseline.py` | Split integrity and actual Flower training/evaluation message handlers |

## Environment

The `.venv` in this folder is prepared. Use its executable directly in PowerShell;
activation is optional. To install again in a fresh checkout with Python 3.11:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

CPU execution is intentional. Images become RGB 64×64 tensors, normalized to
[-1, 1]. Each worker uses one PyTorch thread. No augmentation or compression is
enabled. Ray support on Windows is experimental; Flower recommends WSL2 if
native Windows simulation fails. See [Flower simulation documentation](https://flower.ai/docs/framework/how-to-run-simulations.html).

## Data already downloaded

The active dataset is **`img_align_celeba/`** in this project folder, containing
202,599 JPGs. The selected 300 images have been verified as readable 178×218
aligned/cropped CelebA images. The default manifest, `data/subset/manifest.json`,
points directly to this folder. No image copying or archive extraction is needed.

The ten identities, selected filenames, training/validation/test splits, and
client assignments match the earlier preliminary run. Only the image variant
has changed. The old unaligned dataset folders are no longer required.

To rebuild the default manifest:

```powershell
.venv\Scripts\python.exe -m flower_face.prepare_data --images img_align_celeba
```

Run the aligned/cropped baseline:

```powershell
.venv\Scripts\python.exe -m flower_face.run
```

The launcher reads defaults from `pyproject.toml`. It passes absolute manifest
and output paths because Flower runs a packaged copy of the app in a separate
directory. Dataset images are accessed locally and excluded from the app bundle.
The launcher uses the local SuperLink only; no SuperGrid account is needed.
Flower keeps its managed local SuperLink running for later commands. Its state
and logs for this project are under `.flwr/`.
The launcher uses the installed project environment and disables redundant runtime
dependency installation. It also enables UTF-8 output for Windows terminals and
limits Ray's CPU pool to four cores.

For a different number of rounds:

```powershell
.venv\Scripts\python.exe -m flower_face.run --rounds 5
```

## Experiment design

Before longer federated experiments, run these checks from the project folder:

```powershell
# Repeatedly train on one fixed image per identity (10 images total).
.venv\Scripts\python.exe -m flower_face.check_training overfit --steps 500 --lr 0.1

# Train the same CNN centrally on all 200 training images, validating on 40.
.venv\Scripts\python.exe -m flower_face.check_training centralized --epochs 30

# Once centralized training learns, run longer FedAvg using validation only.
.venv\Scripts\python.exe -m flower_face.run --rounds 30 --skip-test
```

The overfit check aims for 100% training accuracy and loss below 0.05; it stops
early on reaching both. If it has not reached the target, inspect its curve before
proceeding. Its learning rate is intentionally higher than the baseline's.
Centralized training uses the project learning rate (0.01) unless overridden with
`--lr`. These are starting settings, not a guarantee of convergence.

Both checks reuse `task.py` and print training accuracy; centralized training also
prints validation accuracy each epoch. Their checkpoints and metric histories go
to `outputs/overfit-<timestamp>/` or `outputs/centralized-<timestamp>/`. Neither
check evaluates the held-out test set. `--skip-test` likewise records `test: null`
for federated runs. Omit that flag only when ready to evaluate chosen settings.

Thirty centralized epochs and thirty one-local-epoch federated rounds each expose
every training image thirty times, but have different optimizer update sequences;
they are a diagnostic comparison rather than identical optimization workloads.

Defaults: seed 42, ten identities, thirty images per identity, three FedAvg
rounds, one local epoch per round, batch size 16, SGD learning rate 0.01.
All four clients participate in training and validation every round.

| Split | Images per identity | Total | Per client |
| --- | ---: | ---: | ---: |
| Training | 20 | 200 | 50 |
| Validation | 4 | 40 | 10 |
| Final test | 6 | 60 | Server only |

Sampling first chooses identities having at least thirty images and then samples
thirty images per identity using a fixed seed. Each identity is mapped to a
contiguous class label 0–9. Images are split before client allocation. Each
client receives five training images and one validation image of every identity:
a balanced, stratified IID baseline. No image appears in two splits or clients.
The same identities intentionally occur in every split for closed-set evaluation.

`list_eval_partition.txt` is retained but unused: this small demo uses its own
seeded stratified split. Results therefore are not official CelebA benchmark
results. Validation is reported after each round; final test images are evaluated
only after training finishes. Do not tune hyperparameters against the final test.
With ten balanced classes, chance accuracy is 10%; three tiny training rounds
primarily check the pipeline and are not a claim of useful recognition accuracy.

Preparation writes a manifest with original identity IDs, label mapping, every
filename, split, client assignment, image variant, seed, and annotation hash.
Keep that manifest fixed for later algorithm comparisons. To change the subset,
pass `--seed`, `--num-identities`, or `--images-per-identity` when preparing it and
update `num-classes` in `pyproject.toml` if needed. There are at most 35 images per
identity in these annotations, so requesting 40–50 per identity will fail.

## Results and checks

Verified on 2026-09-07: the full three-round Flower simulation completed using
the 300 selected **aligned/cropped images** from `img_align_celeba/`. Every round
received four training replies and four validation replies with zero failed
replies. The saved checkpoint was reloaded successfully and all weights were
finite. All 202,599 dataset filenames match the identity annotation filenames.

| Round | Training loss | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: |
| 1 | 2.305727 | 2.305064 | 7.5% |
| 2 | 2.305361 | 2.304886 | 5.0% |
| 3 | 2.305000 | 2.304618 | 5.0% |

Final test: loss **2.3044**, accuracy **11.67%** (7/60 images; near chance).
This verifies the pipeline, not useful recognition performance.
Aligned results: `outputs/20260907T132745877243Z/`;
Flower run ID: `5508292209555316302`.

The earlier unaligned run is retained in `outputs/20260907T125726511364Z/`
as a historical pipeline check (15% final validation, 10% test accuracy).
Its manifest references the retired dataset and is not used by the default run.
The four existing baseline tests passed during the initial implementation;
this dataset switch was verified with the full aligned-data simulation above.

Native Windows Ray printed access-violation stack traces in worker shutdown code
despite all rounds completing and results being saved. This remains a runtime
limitation; use WSL2/Linux for stable thesis experiments if it persists. No library
code was patched or errors hidden. An earlier synthetic runtime check is separately
labelled `synthetic-runtime-check` in its output manifest and metrics.

Each completed run creates a timestamped directory under `outputs/` containing:

- `final_model.pt`: state dictionary, original identity IDs, and run configuration.
- `metrics.json`: round training/validation metrics, final test metrics, model size,
  image variant, identity IDs, package versions, run configuration, and manifest hash.
- `manifest.json`: a copy of the exact selected images and split assignments.

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Later thesis work

Keep future codecs in a separate, Flower-independent `compression/` package.
Implement and unit-test QSGD, LLZ-p (including exact lossless `p=0`), and LLZ-SI
before integrating them into client/server messaging. Read the source papers in
`papers/` when those implementations begin; no algorithm details are assumed here.
Then compare uncompressed, QSGD, QSGD + LLZ-p, LLZ-SI, and Top-k with matched
data, initialization, and training settings, extending to non-IID partitions.

This baseline does **not** measure actual transmitted bits. Model parameter counts
are not network traffic: later accounting must include serialization, metadata,
and both transmission directions with a documented measurement boundary.
