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
| `flower_face/updates.py` | Adapt model deltas to named QSGD byte packets and validate incoming updates |
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

## Environment

The `.venv` in this folder is prepared. Use its executable directly in PowerShell;
activation is optional. To install again in a fresh checkout with Python 3.11:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

CPU execution is intentional. Images become RGB 64×64 tensors, normalized to
[-1, 1]. Each worker uses one PyTorch thread. No augmentation is enabled.
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
for federated runs. Test evaluation is now disabled by default; use
`--evaluate-test` explicitly when ready to evaluate chosen settings.

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
.venv\Scripts\python.exe -m flower_face.check_training centralized --epochs 100 --lr 0.01
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
ConfigRecord naming the codec, server round, and levels, and the usual training
metrics. No full model or decoded update is included in that reply. Bytes in a
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
clients/rounds and from training. Multi-worker execution can still introduce
floating-point variation between independent runs.

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
