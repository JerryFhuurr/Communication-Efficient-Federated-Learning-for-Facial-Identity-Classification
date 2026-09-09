# Communication-Efficient Federated Learning for Facial Identity Classification
## Flower + PyTorch CelebA baseline

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
| `flower_face/server_app.py` | Run ordinary FedAvg with validation checkpoint selection and optional test evaluation |
| `flower_face/communication.py` | Measure serialized Flower object sizes and log training/validation upload and download totals |
| `compression/qsgd.py` | Standalone L2 QSGD quantizer and dense byte codec; no Flower or PyTorch imports |
| `compression/llz_p.py` | Standalone LLZ-p symbol codec following the supplied paper |
| `compression/qsgd_llz.py` | Compose QSGD quantization with LLZ-p packet encoding |
| `federated_compression/updates.py` | Reusable PyTorch/Flower adapter for named compressed model deltas; no dataset or model imports |
| `flower_face/updates.py` | Backward-compatible import shim for earlier project code |
| `flower_face/reproducibility.py` | Stable model/source hashes and recorded deterministic settings |
| `flower_face/study.py` | Replay checks, paired seed experiments, and aggregate study summaries |
| `compression/check_qsgd.py` | Synthetic codec size/error check with a saved JSON report |
| `flower_face/experiment.py` | Shared checkpoint saving, experiment metadata, and incremental history; no Flower imports |
| `flower_face/check_training.py` | Centralized and tiny-batch training diagnostics using the same CNN |
| `flower_face/prepare_data.py` | Reproducible identity selection and disjoint image splits |
| `flower_face/run.py` | Thin wrapper around `flwr run . local --stream`; supplies absolute local paths and four clients |
| `pyproject.toml` | Dependencies, Flower entry points, experiment defaults |
| `tests/test_baseline.py` | Split integrity and actual Flower training/evaluation message handlers |
| `tests/test_checkpoints.py` | Checkpoint/round matching, tie handling, and isolation of test evaluation |
| `tests/test_communication.py` | Serialization round trip, byte counts, per-recipient accounting, and partial failures |
| `tests/test_qsgd.py` | Quantization statistics, binary round trips, exact sizes, edge cases, and malformed input |
| `tests/test_qsgd_flower.py` | Client/server integration, weighted delta aggregation, serialization, and codec accounting |
| `tests/test_reproducibility.py` | Reply-order invariance, replay guards, paired statistics, and source snapshots |

## Environment

The `.venv` in this folder is prepared. Use its executable directly in PowerShell;
activation is optional. To install again in a fresh checkout with Python 3.11:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

CPU execution is intentional. Images become RGB 64×64 tensors, normalized to
[-1, 1]. Each worker uses one PyTorch thread. Training uses horizontal flips.
Compression is optional and disabled by default. Ray support on Windows is experimental; Flower recommends WSL2 if
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

## Framework boundary

The CelebA code is a replaceable task used to exercise the federated and
compression pipeline. The project is split into three layers:

1. `compression/` implements byte codecs over NumPy arrays and integer symbols.
   It does not import Flower, PyTorch, or the face dataset.
2. `federated_compression/` converts named PyTorch model deltas to and from
   Flower `ConfigRecord` packets. It has no dependency on `flower_face`.
3. `flower_face/` supplies the demonstration model, loaders, client handlers,
   FedAvg strategy, checkpoints, and experiment records.

To reuse the framework for another dataset, replace the model and data functions
in `flower_face/task.py` (or create another task package) while preserving named
floating state dictionaries and the client/server adapter calls. QSGD and LLZ do
not need to change.

Available upload modes are `none`, `qsgd`, and `qsgd-llz`. QSGD is applied to
each client's model delta relative to that round's global model. With
`qsgd-llz`, the same quantized integer codes are encoded by LLZ-p. The integrated
path currently fixes `p=0`, so QSGD and QSGD+LLZ must produce identical decoded
updates, model hashes, and learning metrics; only their packet sizes may differ.

```powershell
.venv\Scripts\python.exe -m flower_face.run --rounds 3 --compression qsgd --skip-test
.venv\Scripts\python.exe -m flower_face.run --rounds 3 --compression qsgd-llz --qsgd-levels 127 --llz-window 128 --skip-test
.venv\Scripts\python.exe -m flower_face.check_llz --rounds 3 --seed 42 --qsgd-levels 127 --llz-window 128
```

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
Centralized training uses the project learning rate (0.1) unless overridden with
`--lr`. These are starting settings, not a guarantee of convergence.

Both checks reuse `task.py` and print training accuracy; centralized training also
prints validation accuracy each epoch. Their checkpoints and metric histories go
to `outputs/overfit-<timestamp>/` or `outputs/centralized-<timestamp>/`. Neither
check evaluates the held-out test set. `--skip-test` likewise records `test: null`
for federated runs. Test evaluation is now disabled by default; use
`--evaluate-test` explicitly when ready to evaluate chosen settings.

Thirty centralized epochs and thirty one-local-epoch federated rounds each expose
every training image thirty times, but have different optimizer update sequences;
they are a diagnostic comparison rather than identical optimization workloads.

Defaults: seed 42, ten identities, thirty images per identity, three FedAvg
rounds, one local epoch per round, batch size 16, SGD learning rate 0.1,
zero weight decay, and training-only horizontal flips.
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

Checkpoint selection and experiment logging were added on 2026-09-08. New runs
save both the lowest-validation-loss model and the final model. Historical runs
below used the earlier final-checkpoint-only implementation.

Verification: eight automated tests passed, including actual FedAvg aggregation
with controlled replies where the best and final rounds differ, tie handling,
and optional test evaluation of only the selected model. Short centralized and
overfit checks passed. A real four-client, three-round simulation completed with
zero failed client replies and the test skipped. Its artifacts are in
`outputs/fedavg-20260908T094207104136Z/`. The saved best model reproduced its
recorded validation metrics when reloaded.

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

## Checkpoint selection and experiment records

The selection rule is fixed before training: **lowest validation loss**, with the
earlier epoch/round retained on an exact tie. Accuracy is recorded but does not
break ties or override loss. Round zero is not a candidate. No early stopping or
learning-rate change is introduced: every requested epoch/round still runs.

The Flower strategy remains standard sample-weighted FedAvg. A small subclass
records the aggregated model after training and selects it using the subsequent
sample-weighted validation loss. Both phases must return successful replies from
all four distinct clients; an incomplete round stops the baseline run.

New federated runs use `outputs/fedavg-<UTC timestamp>/`. Centralized runs retain
their `centralized-` prefix. The directory is created at the start and contains:

- `best_model.pt`: weights from the selected validation epoch/round, including its
  metrics, step number, identity mapping, configuration, and manifest hash.
- `best_checkpoint.json`: a short description of the selected checkpoint.
- `final_model.pt`: weights from the last completed step of a successful run.
- `history.jsonl`: one JSON entry per completed epoch/round, written immediately;
  federated records include successful client counts and whether a new best was saved.
- `experiment.json`: configuration, package/Python versions, platform, split counts,
  selection rule, elapsed time, and completion status.
- `metrics.json`: full completed-run metrics, selected checkpoint information, and
  optional held-out test result (null when skipped).
- `manifest.json`: a copy of the exact selected images and split assignments.

Federated `train_loss` is the sample-weighted average loss during client updates;
it is not a fresh evaluation of the aggregated global model. Centralized
`train_loss` is evaluated after the epoch; `optimization_loss` records the loss
during its updates. Validation loss always evaluates the model being considered
for checkpoint selection. This distinction matters when comparing learning curves.

Best checkpoints and JSON summaries use temporary-file replacement. If a run
is interrupted, completed history entries and the latest saved best checkpoint
remain available; the experiment status stays `incomplete`. These are model
checkpoints, not a full optimizer/RNG resume facility.

The tiny-batch overfit check has no validation set, so it saves only a final model
and logs, without declaring a best validation checkpoint. Centralized training
always skips the test set. An effective `--lr` override is stored in its config.

For ordinary development:

```powershell
.venv\Scripts\python.exe -m flower_face.run --rounds 300
.venv\Scripts\python.exe -m flower_face.check_training centralized --epochs 100 --lr 0.1
```

Only after the experiment settings and selection protocol have been chosen:

```powershell
.venv\Scripts\python.exe -m flower_face.run --rounds 300 --evaluate-test
```

This starts a fresh run, selects a model using validation alone, then evaluates
**that selected model once** on the held-out test set. Its test record names the
checkpoint and round. `--skip-test` remains supported and is mutually exclusive
with `--evaluate-test`. Earlier runs cannot acquire best checkpoints retroactively:
their intermediate weights were never saved.

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Later thesis work

The standalone QSGD quantizer and dense binary codec are now in `compression/`.
See the [codec specification and checks](compression/README.md). Flower can use it
for client delta uploads with `--compression qsgd`; the default remains uncompressed.
Compare using matched settings and the same communication boundary, then repeat
across seeds before drawing performance conclusions. Implement and unit-test LLZ-p (including
exact lossless `p=0`) and LLZ-SI before integrating them. Read the source papers in
`papers/` when those implementations begin; no algorithm details are assumed here.
Then compare uncompressed, QSGD, QSGD + LLZ-p, LLZ-SI, and Top-k with matched
data, initialization, and training settings, extending to non-IID partitions.

The baseline now measures logical serialized message sizes at the boundary
described below. Actual network traffic still needs transport instrumentation;
do not label these counters as packet-level transmitted bits in thesis results.

## Communication accounting

New federated runs automatically measure messages in both directions, separately
for training and validation. FedAvg, model updates, checkpoint selection, and the
data split are unchanged. The instrumentation adds no client/server messages.

The measurement boundary is **a full Flower Message object graph per logical
recipient**, observed at the server's Grid interface. The project is pinned to
Flower 1.36.0. Outgoing instructions are measured after Grid dispatch returns
(or raises), when Flower has populated routing metadata; incoming replies are
measured before aggregation can change their arrays. A submitted instruction is
counted even if no reply arrives; its delivery is not assumed. Error replies are
counted too. The final held-out test runs locally on the server and adds no
federated message cost.

Three sizes are reported, all as integer bytes:

| Field | Meaning |
| --- | --- |
| `raw_array_bytes` | Numeric storage: sum of shape elements × dtype size for every named array |
| `array_data_bytes` | Actual encoded array buffers: `len(Array.data)`, including NumPy serialization headers |
| `codec_payload_bytes` | Complete QSD1 byte packets in compressed uploads, including codec headers and padding |
| `serialized_object_bytes` | Actual byte lengths from Flower's `deflate()` for the Message and every unique descendant object |
| `serialized_object_bits` | Exactly eight times `serialized_object_bytes` |

Flower's `deflate()` here means object serialization, not a compression algorithm
we have added. The object graph includes array chunks, shapes and dtypes, names,
configuration, metrics, message metadata, object headers, and child references.
Measuring the Message alone would miss its model data, which lives in descendant
objects. See Flower's [Message serialization API](https://flower.ai/docs/framework/1.33/en/ref-api/flwr.app.Message.html)
for the object/children interface; the implementation uses the installed 1.36.0 source.

Identical objects are counted once **within each message**, matching Flower's
object-ID representation. The full graph is counted again for each recipient,
phase, and round. No cross-message or runtime cache savings are assumed. The raw
and array-buffer counters count every named array, including duplicates; because
serialized graphs can share identical objects, subtracting the raw counter from
the serialized counter is not a reliable measure of metadata overhead. The
per-message log instead breaks serialized bytes down by object type.

These are **logical serialized object sizes, not measured network traffic**.
The counters exclude separate RPC envelopes/object-tree announcements, transport
headers, acknowledgements, polling, retries, and runtime control traffic. They
also exclude app packaging, local dataset reads, and checkpoint files. Runtime
deduplication or caching can reduce transfers, while transport overhead or retries
can increase them; this metric is not a bound on actual network bytes.

For this 89,834-parameter float32 CNN, one raw model is **359,336 bytes**. With four
clients and validation every round, the expected raw-array accounting is:

| Phase | Direction | Messages per round | Raw array bytes per round |
| --- | --- | ---: | ---: |
| Training | Download to clients | 4 | 1,437,344 |
| Training | Upload to server | 4 | 1,437,344 |
| Validation | Download to clients | 4 | 1,437,344 |
| Validation | Upload metrics to server | 4 | 0 |
| Total | Both directions | 16 | 4,312,032 |

Validation requires another model download to every client. Its metric-only
replies contain zero array bytes but still have a nonzero serialized size. Compare
training communication separately from validation communication when studying
compression; the evaluation schedule must be matched across methods.

Every new federated output directory adds:

- `communication_messages.jsonl`: one observation per message, with round, phase,
  direction, client node ID, routing IDs, error flag, sizes, and an object-type
  breakdown. Node IDs identify simulation participants, not dataset identities.
- `communication.json`: the measurement definition, totals, and per-round counts.
  Each summary has total, upload, download, training, and validation breakdowns.
- `history.jsonl`: each completed round now includes `communication` and
  `cumulative_communication`, alongside its validation metrics.
- `metrics.json`: includes the completed run's `communication` summary.
- `experiment.json`: includes the measurement definition with existing version
  and configuration information.

The message log and summary are written after each Grid exchange. If an exchange
raises, submitted messages and replies already yielded by Grid are retained.
Hard termination during an exchange can leave that exchange unrecorded; check
the experiment's completion status before using totals. All counts and summaries
remain server-side and do not add accounting fields to transmitted payloads.
Measurement and disk logging add runtime overhead, so wall-clock times from old
uninstrumented runs are not directly comparable.

Use the same commands; accounting is enabled automatically:

```powershell
# Short check of all four message paths.
.venv\Scripts\python.exe -m flower_face.run --rounds 3

# A fresh baseline experiment with recorded communication costs.
.venv\Scripts\python.exe -m flower_face.run --rounds 300
```

Previous experiment folders are preserved. Their exact serialized counts cannot
be recovered from parameter counts or saved checkpoints alone. Future compressed
runs must also define their codec payload/metadata format independently of Flower
and retain the same logical-message measurement boundary for fair comparisons.

Verified on 2026-09-08: all 13 automated tests passed, including complete message
reconstruction through Flower's serializer, shared-array deduplication, metadata
costs, repeated recipients/rounds, and partial-error logging. A real three-round
CelebA run completed with four successful training and validation replies each
round, 48 logged messages, matching per-message/round/cumulative totals, and the
test skipped. Results: `outputs/fedavg-20260908T100652995585Z/` (Flower run
`9923936832386669839`). The existing native Windows Ray shutdown traces remain.

| Measured serialized object bytes | Per round in this check | Three-round total |
| --- | ---: | ---: |
| Training, both directions | 2,915,706 | 8,747,118 |
| Validation, both directions | 1,459,342 | 4,378,026 |
| Total | 4,375,048 | 13,125,144 |

The total is 105,001,152 serialized object bits across the three rounds. Byte
counts may vary with routing metadata, serialization, or object sharing; these
observed sizes are not hard-coded into the measurement. The saved best and final
checkpoints both belong to round three in this short pipeline check.

## QSGD client uploads

QSGD is opt-in; default commands and `--compression none` retain the original
uncompressed FedAvg path. To run the integrated version:

```powershell
# Four-client smoke check, s=127.
.venv\Scripts\python.exe -m flower_face.run --rounds 3 --compression qsgd --qsgd-levels 127

# Full development comparison, with the same subset/seed and test still skipped.
.venv\Scripts\python.exe -m flower_face.run --rounds 300 --compression qsgd --qsgd-levels 127
.venv\Scripts\python.exe -m flower_face.run --rounds 300 --compression none
```

Every client receives full-precision global weights `w_t`, performs the same
local SGD training, and computes `delta_i = local_weights_i - w_t`. Each named
tensor is quantized independently with an L2 norm and serialized as a QSD1
packet. The default `s=127` uses seven magnitude bits plus one sign bit per
coordinate, with additional packet metadata. There is no error feedback.

Uploads carry `qsgd: ConfigRecord({tensor_name: packet_bytes})`, plus an `update`
ConfigRecord naming the codec, server round, and levels, the usual training
metrics, and a `client` ConfigRecord with the dataset partition ID. No full model
or decoded update is included in that reply. Bytes in a
ConfigRecord follow Flower's [custom-message approach](https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html).

The server validates the round, codec, levels, exact tensor names, shapes,
dtypes, finite loss/decoded values, and positive sample counts. It decodes all
updates, uses Flower's sample-weighted aggregation on the decoded deltas, and
sets `w_(t+1) = w_t + sum_i(n_i * decoded_delta_i) / sum_i(n_i)`.
The reference is captured afresh each round. All four distinct clients must
reply successfully. Malformed or mixed upload formats stop the experiment.

Training loss describes the client's local SGD updates before compression;
validation evaluates the reconstructed global model. Checkpoint selection still
uses the lowest validation loss. Training and validation downloads remain full
precision, and validation uploads remain metrics-only. QSGD is applied to local
model deltas, not individual minibatch gradients; this is an experimental FedAvg
adaptation, not a reproduction of the paper's entire parallel-SGD algorithm.

Codec randomness uses its own NumPy Generator initialized from
`SeedSequence([seed, server_round, partition_id, 0x51534744])`, visiting tensor
names in sorted order. It does not consume PyTorch or global NumPy RNG state.
The seed, round, and partition ID separate the streams from those of other
clients/rounds and from training. New runs sort replies by dataset partition ID
before both training and validation aggregation. See the reproducibility checks below.

QSGD experiments use `outputs/fedavg-qsgd-<timestamp>/`. Configuration records
`compression` and `qsgd-levels`; experiment metadata documents the update target,
normalization, RNG, and absence of error feedback. Existing best/final
checkpoints and per-round metrics are saved as before.

Communication schema version 2 adds `codec_payload_bytes`, the sum of complete
QSD1 packets. Compressed uploads contain no ArrayRecord, so their `raw_array_bytes`
and `array_data_bytes` are zero; that does not mean zero communication. Compare
**`serialized_object_bytes`/`serialized_object_bits`** across methods: these
continue to count the full transmitted logical object graph, including the byte
packets, tensor names, codec metadata, metrics, and Flower object headers. The
Grid logs each encoded reply before any decoding. Decoded aggregation records
are server-local copies and are never counted as additional uploads.

The logical measurement boundary is unchanged from schema version 1; the new
packet sub-counter is zero for uncompressed messages. It is a subset of serialized
bytes and must not be added to them. These counters still exclude transport
traffic, cache effects, and retries. Upload compression affects only one of the
three model transfers per client per round, so total savings will be smaller than
the codec's roughly fourfold reduction at `s=127`.

Use matched manifest hashes, model/training configuration, round counts, and
validation schedules when comparing. Select each model by the same validation
rule, keep the held-out test out of development, and repeat across seeds before
drawing thesis conclusions. The original baseline output directories remain valid
references and are not rewritten by QSGD runs.

Verification: all 65 tests passed, including real ClientApp/strategy dispatch
through Flower serialization for both modes, unequal client sample counts,
positive and negative updates, reference refresh over two rounds, rejected
malformed updates, isolated codec RNG, and accounting of compressed bytes before
decoding. The standalone compression package still imports no Flower or PyTorch.

The real three-round check completed at
`outputs/fedavg-qsgd-20260908T104355354317Z/` with 48 messages, finite best/final
checkpoints, and the test skipped. Each training upload carried 90,078 bytes of
QSD1 packets. Complete serialized traffic across all three rounds was 9,844,656
bytes, versus 13,125,144 bytes in the earlier uncompressed three-round check.
The native Windows Ray shutdown traces remain; all client replies succeeded.

### First 300-round QSGD comparison

Completed QSGD run: `outputs/fedavg-qsgd-20260908T104659729947Z/`, Flower run
`12624072430969573136`. Baseline: `outputs/fedavg-20260908T101443098167Z/`.
The manifest hash, package versions, and model/training settings matched. All
4,800 QSGD messages were verified against phase, round, cumulative, and final
totals. Both QSGD checkpoints contain finite weights; reloading the selected model
reproduced its recorded validation loss and accuracy. The held-out test was skipped.

| Result | Uncompressed FedAvg | QSGD delta uploads, s=127 |
| --- | ---: | ---: |
| Selected round (lowest validation loss) | 207 | 207 |
| Selected validation loss | 1.747309 | 1.729523 |
| Selected validation accuracy | 42.5% | 40.0% |
| Final round-300 validation accuracy | 42.5% | 47.5% |
| Peak validation accuracy during run | 50.0% | 52.5% |
| Training upload serialized bytes | 437,382,000 | 109,341,348 |
| Total serialized bytes, including validation | 1,312,507,896 | 984,470,844 |

Training upload bytes decreased by 75.00%; total serialized bytes decreased by
24.99%. Downloads were full precision in both runs; small byte differences in
routing metadata are included in these observed totals. This is one seed with
only 40 validation images, so the accuracy/loss differences do not establish
superiority or equivalence. The chosen models follow the loss rule, not the peak
accuracy row. Detailed checks and comparison are in
`outputs/fedavg-qsgd-20260908T104659729947Z/baseline_comparison.json`.

## Reproducibility and paired seed studies

New runs use protocol `partition-order-v1`. Flower's original reduction adds
updates in reply-list order, which can vary with scheduling. Floating-point sums
can differ when that order changes. A controlled test demonstrated this effect;
sorting by stable dataset partition IDs fixes this source of variation. Sorting
by runtime node ID would not fix it because those IDs can change between runs.

Every training and validation reply now includes a `client` ConfigRecord with
`partition-id`. The server requires all partitions exactly once, rejects missing,
duplicate, or changing partition identities, and aggregates in ascending partition
order. This preserves the mathematical sample-weighted FedAvg rule, but can change
floating-point results relative to old runs. Communication schema version 3 records
`client_partition_id`; the additional transmitted metadata is counted in both modes.
Use fresh runs of both methods under this protocol for paired comparisons.

Seeding also enables `torch.use_deterministic_algorithms(True)`, uses one PyTorch
thread, disables cuDNN benchmarking, and requests deterministic cuDNN algorithms.
Unsupported nondeterministic operations raise rather than silently proceeding.
The project currently trains on CPU. These controls follow
[PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html);
they do not promise bitwise agreement across hardware, library versions, or platforms.

`experiment.json` records actual deterministic settings, an initial-model hash,
and a source hash. Python files are hashed byte-for-byte; `pyproject.toml` is hashed
by parsed values because Flower reformats TOML during packaging. Each completed
federated round records `model_sha256`. Model hashes cover sorted tensor names,
shapes, dtypes, and raw values, excluding checkpoint container timestamps and paths.

`--seed` changes training initialization, local data order, and codec randomness.
It does **not** regenerate the subset or alter train/validation/test membership.
The manifest's original preparation seed and filename assignments remain fixed.

```powershell
.venv\Scripts\python.exe -m flower_face.run --rounds 300 --seed 43 --compression none
.venv\Scripts\python.exe -m flower_face.run --rounds 300 --seed 43 --compression qsgd --qsgd-levels 127
```

The study runner performs the reproducibility gate and paired comparisons:

```powershell
.venv\Scripts\python.exe -m flower_face.study --seeds 42 43 44 --rounds 300 --repeat-rounds 3
```

It first runs each method twice for three rounds at the first seed, requiring
identical model hashes, training/validation metrics, and selected-checkpoint
metadata. Elapsed times, routing IDs, and serialized byte totals are excluded from
exact replay comparisons: those can differ without changing the learned model.
A replay failure stops the study before the long experiments. Passing a short
replay is evidence for that configuration and duration, not a universal guarantee.

After the gate, it runs each seed with uncompressed FedAvg and QSGD (`s=127` by
default), alternating which method goes first across seed pairs. Runs are sequential
and share the fixed manifest. Paired runs must have the same initialization hash,
source, manifest hash, package versions, platform, training settings, and selection
rule. The study always skips the held-out test. Replay checks are excluded from
the seed statistics.

Each study creates `outputs/study-<timestamp>/` containing:

- `app/`: a snapshot of Python source and `pyproject.toml`, excluding images,
  virtual environments, and outputs. Flower bundles this small directory for all
  runs instead of repeatedly scanning the full project. Data stays in its original
  local folder and is accessed through the absolute manifest path.
- `replay-<method>-<repeat>/`: isolated check outputs and `terminal.log`.
- `seed-<seed>-<method>/`: the full experiment artifacts and `terminal.log` for
  each seed/method pair, with checkpoints, histories, and communication records.
- `study.json`: settings, fingerprints, replay results, completed run records,
  and progress/status; written after each completed run.
- `summary.json`: means and **sample standard deviations** across seeds, plus
  paired accuracy/loss differences and communication reductions for each seed.

Study collection checks completion, client participation, communication totals,
configuration, source/manifest fingerprints, and checkpoint hashes. A failed run
stops the study and retains its logs; interrupted studies are labelled accordingly.
There is no automatic resume. Individual `run` commands also accept `--output-dir`;
each launch now writes a unique override TOML file to avoid shared-config overwrites.

Checkpoint/JSON replacement retries brief Windows sharing/access locks up to
eight attempts (under one second of total backoff), then raises if the error
persists. This covers transient file readers or scanners while keeping atomic
replacement. Study collection checks experiment completion even when Flower's
streaming CLI exits with code zero after an application error; partial runs never
enter the summary.

Three training seeds on this fixed 40-image validation set are a preliminary
robustness check. The reported standard deviation is variability across training
seeds, not a confidence interval or a measure of dataset-sampling uncertainty.
Report selected validation accuracy/loss together; do not replace the fixed
minimum-loss selection rule with each run's peak accuracy after seeing results.

### Completed paired study: seeds 42, 43, 44

Results are saved in `outputs/study-20260908T114723972670Z/summary.json`, with
full run records in `study.json`. All six 300-round runs completed. Each pair
had the same initial-model hash, and the three seeds had different initial
models. Both three-round replay checks matched model hashes and learning metrics
exactly. Across the six long runs, all 28,800 recorded messages had successful
replies where applicable; byte totals reconciled and checkpoint hashes matched
their recorded rounds. The held-out test was skipped throughout.

| Training seed | Selected round, none / QSGD | Validation accuracy, none / QSGD | Validation loss, none / QSGD |
| --- | ---: | ---: | ---: |
| 42 | 207 / 207 | 40.0% / 45.0% | 1.728491 / 1.721598 |
| 43 | 300 / 300 | 17.5% / 17.5% | 2.132865 / 2.133377 |
| 44 | 252 / 252 | 40.0% / 40.0% | 1.897066 / 1.914543 |

| Across three training seeds | Uncompressed FedAvg | QSGD, s=127 |
| --- | ---: | ---: |
| Selected validation accuracy, mean +/- sample SD | 32.50% +/- 12.99 pp | 34.17% +/- 14.65 pp |
| Selected validation loss, mean +/- sample SD | 1.919474 +/- 0.203116 | 1.923172 +/- 0.206025 |
| Mean training upload serialized bytes per run | 437,598,600 | 109,557,048 |
| Mean total serialized bytes per run | 1,312,943,496 | 984,902,844 |

QSGD reduced training uploads by approximately **74.96%** and total counted
serialized bytes by **24.99%** in each pair. These are logical serialized-object
measurements including metadata, not captured network traffic. The paired selected
accuracy difference was +1.67 percentage points on average (sample SD 2.89 pp).
The small fixed validation set and large variation between training seeds do not
establish an accuracy advantage or equivalence. Seed 43's selected checkpoint was
the last round in both modes, so this experiment also does not establish convergence.

Verification: **78 tests passed**, including arrival-order and runtime-node-ID
permutations, replay mismatch detection, source snapshots, paired summaries, and
transient Windows file-lock handling. Native Windows Ray access-violation traces
still appeared in terminal logs, despite every study run completing successfully.
The earlier failed development studies are retained separately and excluded from
these results.

## Standalone QSGD + LLZ-p codec

The LLZ-p codec now follows the supplied revised manuscript's Section III and
Algorithms 2/3. Source fingerprints, a reading guide, notation corrections, and
the complete packet format are documented in `papers/README.md`. The source PDFs
remain in their original local folder. The revised manuscript clarifies several
indexing and decoding details from the older conference paper.

- `compression/llz_p.py`: encode/decode signed integer symbols using the
  reconstructed sliding-window dictionary; `parse` exposes inspectable triplets.
- `compression/qsgd_llz.py`: encode already quantized tensors and preserve the
  QSGD norm, levels, shape, and float dtype in a complete packet.
- `compression/check_llz.py`: compare both packet formats on exactly the same
  QSGD codes and save inputs, encoded packets, fingerprints, and measurements.
- `tests/test_llz_p.py`: paper example, exhaustive reference comparisons,
  lossless recovery, bounded lossy error, malformed packets, and byte accounting.

`p` is an integer tolerance in signed quantization levels. At `p=0`, LLZ adds no
error: recovered codes and QSGD-reconstructed floating-point bytes match exactly.
At `p=1`, each code can differ by at most one level. The revised paper uses
`s-1` intervals; its `s=128` corresponds to our existing `levels=127`. The
QSGD quantizer and Flower training configuration have not changed.

Run the standalone verification and comparison:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_llz_p.py -q
.venv\Scripts\python.exe -m compression.check_llz --levels 127 --p 0 1 --window-size 128
```

The full default check (levels 3 and 127, p 0 and 1) completed at
`outputs/llz-codec-20260908T123533255799Z/`. The table below shows the lossless
second stage at `levels=127`, window 128, seed 42, and 4,096 float32 inputs:

| Synthetic input | QSGD bytes | QSGD + LLZ-p, p=0 bytes | Additional savings |
| --- | ---: | ---: | ---: |
| Gaussian | 4,116 | 3,185 | 22.62% |
| Gaussian with approximately 90% zeros | 4,116 | 1,039 | 74.76% |
| Repeated pattern before stochastic quantization | 4,116 | 1,206 | 70.70% |
| All zeros | 4,116 | 154 | 96.26% |

These are complete standalone packet sizes, including QSGD metadata, LLZ headers,
and padding. They are synthetic codec checks, not measurements of client updates
or Flower/network traffic. Incompressible inputs can expand; the codec has no
hidden fallback. The report also records the extra distortion for positive `p`.

All **142 tests passed**, including 64 new LLZ tests. The manuscript's worked
example reproduced its exact triplets and reconstructed sequence. The standalone
modules import neither Flower nor PyTorch. The next integration step is to add
`p=0` LLZ packets to Flower and verify identical model hashes against QSGD alone
before comparing communication or testing lossy settings in training. LLZ-SI and
error feedback remain future work; no federated experiment used LLZ in this stage.

## Flower integration: lossless LLZ after QSGD

`--compression qsgd-llz` now sends QLP1 packets containing QSGD-quantized model
deltas followed by LLZ-p. Flower currently requires **`llz-p=0`**. Positive
tolerances remain available in the standalone codec check, but are rejected in
Flower until the lossless integration has been evaluated. Defaults are still
uncompressed FedAvg; the new configuration entries are `llz-p=0` and
`llz-window=128`.

Both QSGD methods draw the same quantized symbols from the same isolated random
stream, in sorted tensor-name order. LLZ encoding draws no additional randomness.
Each tensor has its own dictionary. The server validates the codec tag, round,
levels, LLZ tolerance/window in both the envelope and packet, tensor names,
shapes, dtypes, and client sample counts before aggregating. Both methods use the
same ordered sample-weighted delta aggregation and full-precision downloads.
The standalone QLP1 decoder accepts optional expected tolerance/window arguments
so the adapter checks these without decoding a packet repeatedly.

The existing `qsgd` ConfigRecord holds byte packets for either method; its name
does not identify their wire format. The `update` record identifies the codec.
Communication schema 4 extends the packet-counter description to include QLP1.
The measurement boundary and arithmetic are unchanged. Full serialized-message
counts include the extra LLZ envelope metadata as well as every nested packet
header and padding bit. Decoded tensors exist only at the server and are not
counted as uploads.

Run a single LLZ experiment:

```powershell
.venv\Scripts\python.exe -m flower_face.run --rounds 10 --seed 42 --compression qsgd-llz --qsgd-levels 127 --llz-p 0 --llz-window 128 --skip-test
```

For an automatically checked pair, use:

```powershell
.venv\Scripts\python.exe -m flower_face.check_llz --rounds 10 --seed 42 --qsgd-levels 127 --llz-window 128
```

`flower_face/check_llz.py` snapshots the application code, runs fresh QSGD and
QSGD+LLZ experiments sequentially, and collects their completed artifacts. It
requires equal source/data/environment fingerprints, configuration other than
compression/output directory, and initialization hashes. It then requires exact
equality of every round's model hash, training and validation metrics, and selected
checkpoint metadata. Only after these checks pass does it report upload and total
serialized-byte reductions. The check always skips the test set.

Outputs are saved in `outputs/llz-flower-<timestamp>/comparison.json`, with each
method's terminal log, checkpoints, history, and message accounting under its own
directory. A failure stops the comparison, records a failed status, and retains
logs. Routing IDs, timestamps, and byte totals can differ without affecting model
equality; they are measured but excluded from the learning-equivalence check.
Use `--rounds 300` for a longer paired experiment with the same validation rules.

Verification: **178 tests passed**, including real Flower handler dispatch and
serialization for the new method, unequal client sample weighting and reference
refresh across rounds, exact multi-tensor QSGD/LLZ delta agreement, full-message
accounting, incompatible packet rejection, and deliberately mismatched comparison
results. The existing `flower_face.study` remains the uncompressed/QSGD seed study;
use `flower_face.check_llz` for this new QSGD/LLZ pair.

The first real CelebA comparison completed at
`outputs/llz-flower-20260908T125215320610Z/comparison.json`: 10 rounds, four
IID clients, seed 42, QSGD levels 127, LLZ p=0, window 128. Initial model hashes,
every round's model hashes and learning metrics, and selected-checkpoint metadata
matched exactly. Both methods selected round 10 (validation loss 2.303024,
accuracy 12.5%). This short run verifies integration, not convergence or final
recognition quality. The held-out test was skipped in both runs.

| Complete serialized-object counts, 10 rounds | QSGD | QSGD + LLZ-p=0 |
| --- | ---: | ---: |
| Training upload bytes | 3,651,860 | 1,097,386 |
| Training download bytes | 14,577,540 | 14,577,530 |
| Validation upload + download bytes | 14,600,560 | 14,600,540 |
| Total bytes | 32,829,960 | 30,275,456 |
| Messages | 160 | 160 |
| Error replies | 0 | 0 |

The additional reduction versus QSGD alone was **69.95% for training uploads**
and **7.78% for total serialized bytes**. Total savings are smaller because
training and validation downloads still carry full-precision models. The small
download/validation byte differences come from message metadata; model values
were identical. All 320 message records reconcile with their phase and final
totals. Native Windows Ray shutdown traces still appeared, although both runs
completed successfully. No claim of captured network-traffic savings is made.

## Three-seed 300-round lossless LLZ comparison

Seeds **42, 43, and 44** have now completed paired QSGD and QSGD+LLZ-p=0
experiments: six runs of 300 rounds, using the unchanged small CNN, four IID
clients, QSGD levels 127, and LLZ window 128. All runs share the same source,
dataset manifest, training configuration (apart from seed/method/output path),
and recorded environment. Each seed uses a distinct model initialization on the
same fixed data split.

| Seed | Selected round | Validation accuracy, both methods | Training upload saved | Total serialized bytes saved |
| --- | ---: | ---: | ---: | ---: |
| 42 | 207 | 45.0% | 71.36% | 7.94% |
| 43 | 300 | 17.5% | 72.51% | 8.07% |
| 44 | 252 | 40.0% | 69.19% | 7.70% |

Checkpoint selection uses the lowest validation loss, rather than the highest
validation accuracy. Mean selected-checkpoint validation accuracy is **34.17%**
with sample standard deviation **14.65 percentage points** for both methods.
Mean upload reduction is **71.02%** (sample SD 1.69 percentage points); mean total
reduction is **7.90%** (sample SD 0.19 percentage points).

| Seed | QSGD upload bytes | LLZ upload bytes | QSGD total bytes | LLZ total bytes |
| --- | ---: | ---: | ---: | ---: |
| 42 | 109,556,748 | 31,380,383 | 984,901,644 | 906,728,879 |
| 43 | 109,558,548 | 30,117,574 | 984,908,844 | 905,462,470 |
| 44 | 109,556,748 | 33,756,207 | 984,901,644 | 909,102,903 |
| Sum | 328,672,044 | 95,254,164 | 2,954,712,132 | 2,721,294,252 |

The saved-artifact audit verified exact model hashes and training/validation
metrics across **all 900 paired rounds**, plus identical selected checkpoints
within each pair. All **28,800 message records** reconcile with their per-phase
and final counts, with zero error replies in the six completed runs. The first
seed-43 attempt stopped during round 1 with a Flower runtime task-token
authentication failure; it was excluded and rerun with unchanged settings.

These measurements include codec headers, padding, and Flower record/message
metadata at the logical per-recipient serialization boundary. They are **not
captured network traffic**. Full-precision training and validation downloads
explain why the overall reduction is smaller than the upload reduction. At
`p=0`, LLZ preserves the QSGD output exactly; QSGD itself remains lossy relative
to the original floating-point updates.

This completes the three-seed lossless integration comparison. Recognition
quality remains variable across seeds, and the 40-image validation split is
small. The sample standard deviations describe training-seed variation on one
fixed split, not confidence intervals or performance on new data splits. The
held-out test set was skipped in all six runs.

Artifacts:

- Seed 42: `outputs/llz-flower-20260908T125456428769Z/comparison.json`
- Seed 43: `outputs/llz-flower-20260908T131541417935Z/comparison.json`
- Seed 44: `outputs/llz-flower-20260908T132726021518Z/comparison.json`
- Audited aggregate: `outputs/llz-three-seeds-20260908T132726021518Z/summary.json`
- Audit script: `outputs/llz-three-seeds-20260908T132726021518Z/aggregate.py`
- Excluded failed attempt: `outputs/llz-flower-20260908T130859648340Z/`

The aggregate records source/data fingerprints, input report hashes, per-seed
results, means, sample standard deviations, and exact byte totals. To repeat its
audit while the working source still matches these experiments:

```powershell
.venv\Scripts\python.exe outputs/llz-three-seeds-20260908T132726021518Z/aggregate.py
```

## Training diagnosis: centralized references, seeds 42–44

On 2026-09-09, three new 300-epoch centralized runs were compared with the saved
300-round uncompressed FedAvg and QSGD runs. The CNN, preprocessing, learning
rate (0.01), batch size (16), and fixed split remained unchanged. Initial model
hashes match within each seed. The held-out test set was unused.

| Seed | Selected FedAvg accuracy | Selected QSGD accuracy | Selected centralized accuracy | Centralized selected epoch |
| --- | ---: | ---: | ---: | ---: |
| 42 | 40.0% | 45.0% | 47.5% | 77 |
| 43 | 17.5% | 17.5% | 40.0% | 117 |
| 44 | 40.0% | 40.0% | 32.5% | 60 |
| Mean +/- sample SD | 32.50% +/- 12.99 pp | 34.17% +/- 14.65 pp | 40.00% +/- 7.50 pp | |

All selections use minimum validation loss, not maximum accuracy. LLZ-p=0
shares QSGD's exact learning curves. The six saved federated runs were audited,
and the earlier QSGD curves exactly match the later LLZ study. The new central
runs passed initialization/environment and checkpoint hash/selection checks.
Saved models were also evaluated on the entire training and validation sets.

All three centralized models reached **100% training accuracy**, but their final
validation accuracies were 45.0%, 42.5%, and 45.0%; final validation losses rose
to 6.2391, 5.0620, and 6.7330. This shows overfitting on the small training subset.
Seed 43's final federated training accuracy was only 29.0% for both uncompressed
FedAvg and QSGD, indicating slow fitting in that setting. Compression alone does
not explain its weakness. Centralized selected accuracy did not improve for
every seed, and these results do not establish statistical superiority.

Each method processes **60,000 training examples per seed**. Centralized training
uses 3,900 sequential SGD steps; federated training uses 4,800 local steps summed
across four clients and 300 aggregations. Shuffling, incomplete batches, and
averaging differ. This is an equal-data-exposure diagnostic, not an identical
optimization or wall-time comparison. Historical federated optimization loss
describes changing local models; it is not global training-set loss.

The next proposed experiment is a controlled learning-rate comparison using
uncompressed FedAvg, with the same model, split, seeds and budget. Diagnose
training speed before increasing model size, and investigate regularization or
augmentation if fitting improves while the validation gap persists. No tuning
or model changes were made in this stage.

The complete diagnosis, raw curves, PNG/SVG figures, and new checkpoints are in
`outputs/training-diagnosis-20260909T091351893615Z/`. Start with `report.md` and
`learning-curves.png`; `centralized-generalization.png` shows the train/validation
gap. Reusable runner/plotting commands are documented in `scripts/README.md`.
Plot dependencies are separate from the training configuration. The original
training-package source fingerprint remains unchanged.

## Controlled learning-rate comparison

On 2026-09-09, uncompressed FedAvg was compared at learning rates **0.01, 0.03,
and 0.1** across seeds 42, 43 and 44, with a fixed 300-round budget. The three
audited 0.01 runs were reused and six new runs completed. Each new run uses the
original study's saved application snapshot, matching its baseline's source,
data, initialization and environment; only learning rate and output path differ.

The rule declared before training selects the rate with the lowest mean
validation-selected loss across seeds, with smaller rate breaking exact ties.
Accuracy is reported at those same minimum-loss checkpoints.

| Learning rate | Selected validation loss, mean +/- sample SD | Selected accuracy, mean +/- sample SD | Mean final training accuracy |
| --- | ---: | ---: | ---: |
| 0.01 | 1.9195 +/- 0.2031 | 32.50% +/- 12.99 pp | 60.17% |
| 0.03 | 1.8097 +/- 0.0857 | 30.00% +/- 4.33 pp | 100.00% |
| 0.1 | 1.7279 +/- 0.0903 | 37.50% +/- 5.00 pp | 100.00% |

**0.1 is the preferred candidate among these tested rates.** Its selected
checkpoints occur at rounds 72, 48 and 53, with validation accuracies 42.5%,
32.5% and 37.5%. It improves selected validation loss for all three seeds
relative to 0.01; selected accuracy improves for two seeds and drops by 2.5 pp
for seed 44. Both higher rates overcome the poor training fit of seed 43, but
their late-round validation losses still rise markedly. Improving fitting does
not remove overfitting. The next proposed experiment is a small weight-decay
comparison at rate 0.1, keeping the other controls fixed.

All nine accepted runs passed checkpoint/configuration and communication audits:
2,700 recorded rounds, 43,200 message records, zero error replies. Twelve new
sweep tests passed. The test set was unused, and project defaults remain at
learning rate 0.01. These are exploratory tuning results on one repeatedly used
40-image validation split, not a significance claim or a new held-out estimate.

Run the sweep or regenerate its plots with:

```powershell
.venv\Scripts\python.exe scripts/sweep_learning_rate.py
.venv\Scripts\python.exe scripts/plot_lr_sweep.py outputs/lr-sweep-20260909T093522322142Z
```

The first command launches new training; the second reads saved results only.
The completed artifacts are in `outputs/lr-sweep-20260909T093522322142Z/`,
including `report.md`, `summary.json`, raw `curves.json`, and both PNG/SVG
figures (`learning-rate-curves` and `learning-rate-summary`). The frozen source,
logs and checkpoints are retained. See `scripts/README.md` for details.

## Weight-decay comparison at learning rate 0.1

Weight decay is now configurable through `task.train(..., weight_decay=0.0)`,
Flower's `weight-decay` configuration, and `--weight-decay` on both the Flower
launcher and centralized checker. `flower_face.run` also accepts `--lr`.
PyTorch SGD applies coupled L2 decay to all model parameters, including biases;
reported losses remain cross-entropy without a parameter penalty. Missing
configuration means zero, and the project defaults remain LR 0.01 / decay 0.0.

On 2026-09-09, a controlled comparison tested decay **0, 0.0001 and 0.001** at
LR 0.1 across seeds 42, 43 and 44. All nine runs used the same new application
snapshot and 300 rounds. Before positive decay was tested, the three zero-decay
runs reproduced **all 900 legacy model hashes, learning metrics and selected
checkpoints exactly**, verifying compatibility with the earlier rate-0.1 runs.

| Weight decay | Selected validation loss, mean +/- sample SD | Selected accuracy, mean +/- sample SD | Mean final training accuracy |
| --- | ---: | ---: | ---: |
| 0 | 1.7279 +/- 0.0903 | 37.50% +/- 5.00 pp | 100.00% |
| 0.0001 | 1.7522 +/- 0.0579 | 38.33% +/- 6.29 pp | 100.00% |
| 0.001 | 1.7772 +/- 0.0504 | 37.50% +/- 2.50 pp | 100.00% |

**Zero decay remains preferred under the predefined minimum-mean-validation-loss
rule.** Positive decay changes individual results and can reduce late validation
loss, but neither candidate improves aggregate selected loss. All models still
memorize the training subset. The small accuracy increase at 0.0001 does not
establish a reliable benefit on this repeatedly used 40-image validation set.
This result applies to these tested values and configuration, not weight decay
in general. No test images were loaded.

All nine runs passed audits: 2,700 round records, 43,200 message records and zero
error replies. The 199-test suite passed after optimizer integration; all 12
focused weight-decay tests passed after the three new sweep-control tests were
added. Tests cover zero-decay equivalence, the SGD update equation, validation,
client forwarding and controlled comparison gates.

The completed artifacts are in `outputs/wd-sweep-20260909T100745974156Z/`:
`report.md`, `summary.json`, raw curves, replay records, frozen source,
checkpoints and PNG/SVG figures. The next proposed experiment is training-only
augmentation at LR 0.1 / decay 0, keeping the other controls fixed.

```powershell
# New controlled sweep, including three full zero-decay replay runs.
.venv\Scripts\python.exe scripts/sweep_weight_decay.py

# Plot the completed sweep without retraining.
.venv\Scripts\python.exe scripts/plot_lr_sweep.py outputs/wd-sweep-20260909T100745974156Z --parameter weight-decay

# Single run with explicit optimizer settings.
.venv\Scripts\python.exe -m flower_face.run --rounds 300 --seed 42 --compression none --lr 0.1 --weight-decay 0 --skip-test
```

See `scripts/README.md` for resume support and plotting dependencies. Historical
diagnostic runners that require byte-identical training source reject this new
revision against older snapshots; their stored figures can still be regenerated.

## Training-only horizontal-flip comparison

The final preliminary model-setting test compared no augmentation with
`RandomHorizontalFlip(p=0.5)` across seeds 42, 43 and 44. Runs used uncompressed
FedAvg for 300 rounds at learning rate 0.1 and weight decay 0. Flipping occurs
only after resize while loading training images; validation, test, and checkpoint
evaluation remain unaugmented.

| Training augmentation | Selected validation loss, mean +/- sample SD | Selected accuracy, mean +/- sample SD | Mean final training accuracy |
| --- | ---: | ---: | ---: |
| None | 1.7279 +/- 0.0903 | 37.50% +/- 5.00 pp | 100.00% |
| Horizontal flip | **1.5902 +/- 0.0841** | **53.33% +/- 5.20 pp** | 91.50% |

Horizontal flipping improved selected accuracy for every seed: 42.5% to 57.5%,
32.5% to 55.0%, and 37.5% to 47.5%. Mean selected loss also improved under the
predeclared selection rule. Late validation loss fell substantially, while not
all final models memorized every training image. This supports horizontal
flipping as useful training regularization for the current subset.

Before augmented runs began, three new no-augmentation runs reproduced all 900
legacy model hashes, learning metrics and selected checkpoints exactly. All six
new runs passed artifact audits: 1,800 rounds, 28,800 message records, and zero
error replies. The test set remained unused. These results still come from one
repeatedly used 40-image validation split and are not an unbiased final estimate.

The project training defaults are now frozen for the next stage at learning rate
**0.1**, weight decay **0**, training-only horizontal flipping, four IID clients,
and one local epoch per round. Earlier QSGD and LLZ experiments used learning
rate 0.01 without augmentation, so compression comparisons must be rerun under
this frozen setup before making combined accuracy/communication claims.

Complete outputs are in
`outputs/augmentation-sweep-20260909T105719444213Z/`, including `report.md`, raw
curves, checkpoints, frozen source, and PNG/SVG figures. Reusable commands:

```powershell
.venv\Scripts\python.exe scripts/sweep_augmentation.py
.venv\Scripts\python.exe scripts/plot_augmentation_sweep.py outputs/augmentation-sweep-20260909T105719444213Z
```
