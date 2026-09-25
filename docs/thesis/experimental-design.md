# LLZ-p experimental design v1.0

Prepared 25 September 2026 for the Monday 28 September milestone. This document
records the completed exploratory study and specifies subsequent work. It is
**not retrospective preregistration**. Proposed experiments below were not run
while writing it. Background and citations: [literature review](literature-review.md).

## 1. Objective and research questions

Evaluate LLZ-p as a second compression stage after QSGD for client model updates
in Flower FedAvg. Facial identity classification is the demonstration task.

| Question | Comparison and endpoint | Current evidence |
|---|---|---|
| RQ1: Does p=0 preserve QSGD exactly and reduce upload? | Same seed/configuration; symbol, reconstructed-update, per-round model, metric, and checkpoint equality; serialized upload bits | Exact training replay passed for the 15-round seed-42 fine-tuning pilot |
| RQ2: What is the positive-p accuracy/distortion/communication trade-off? | QSGD versus p=0/p=1; validation loss/accuracy and separately defined QSGD/LLZ errors | p=1 pilot completed; no claim of convergence or generality |
| RQ3: How much upload reduction survives total-message and computation accounting? | Separate training upload, downloads, validation, codec work, and round duration | Bytes measured; dedicated codec and round timing instrumentation remains planned |
| RQ4: Does the trade-off persist across randomness and heterogeneous clients? | Paired seeds first; subsequently a fixed non-IID partition protocol | Stronger-model seed confirmation and non-IID study remain planned |

RQ1 is a correctness requirement, not a statistical hypothesis. RQ2-RQ4 are
empirical questions. No convergence theorem for the current FedAvg implementation
is claimed.

## 2. Frozen reference setting

The completed pilot's authoritative configuration is its saved `study.json`,
not mutable defaults in `pyproject.toml`. The reusable template is
[`digiface-finetune-compression.toml`](../../experiments/digiface-finetune-compression.toml).

| Item | Reference setting |
|---|---|
| Data source | Local DigiFace package `data/subjects_0-1999_72_imgs` |
| Task | 100-way closed-set identity classification; identities 0-99 |
| Subset | 50 selected images per identity: 30 train, 10 validation, 10 test |
| Totals | 3,000 train; 1,000 validation; 1,000 held-out test images |
| Partition | Fixed manifest seed 42; shuffled, stratified image split; approximately IID class-balanced allocation |
| Clients | Four simulated CPU clients; all participate in training and validation each round |
| Per-client data | 750 training and 250 validation images; held-out test is unassigned to clients |
| Model | ImageNet ResNet-18 prefix through layer3 frozen; layer4 and 512-to-100 classifier train |
| Trainable coordinates | 8,445,028 |
| Preprocessing | Resize 256, center crop 224, ImageNet normalization |
| Cache | Layer3 features, shape 256 x 14 x 14, float32; original and horizontal-flip views |
| Augmentation | Random choice of cached original/flip view in training; original view for evaluation |
| Initialization | ImageNet layer4 and the same validation-selected seed-42 head for every method |
| Training | 15 rounds; one local epoch per round; batch size 16 |
| Optimizer | Local SGD, no momentum, no weight decay; rebuilt for each local call |
| Learning rates | Layer4 0.003; classifier 0.03; fixed throughout the run |
| Aggregation | Example-weighted FedAvg; deterministic ascending client partition order |
| QSGD | 127 positive intervals, one L2 norm per tensor, dense sign/magnitude packet |
| LLZ | p=0 and p=1; window 128; reconstructed-history dictionary reset per tensor |
| Feedback | No error feedback or residual state |
| Downloads | Full layer4/head state, uncompressed, for both training and validation |
| Selection | Lowest validation loss; earliest round for exact ties |
| Test evaluation | Disabled for all pilot methods |

The generic preparation code selects the first classes in natural order, not a
random sample of all DigiFace identities. The fixed manifest is the exact record
of selected filenames and assignments. Each client sees the same class set, but
per-class training counts can differ by one because 30 is not divisible by four.
Do not recreate the split for each training seed.

The same identity appears across train/validation/test by design. This is not
subject-disjoint recognition, face verification, or a test of unseen identities.
Exact duplicate content is rejected by preparation; near-duplicates or shared
rendering factors may still limit independence. For a future real dataset,
session/person grouping must follow its intended evaluation task rather than
blindly copying this split.

The warmed head was selected using the existing validation set. Seeds 43/44
would vary subsequent training/quantization randomness, not the data split or
the complete pretraining/model-selection pipeline. The test set must remain
unused for selecting models, hyperparameters, or compression settings.

## 3. Update semantics and fixed buffers

For client k at round t, define `delta[k,t] = local_model[k,t] - global_model[t]`
on the trainable tensors. Compressed methods send a reconstruction of this delta;
the server averages reconstructed deltas using training-example counts and adds
the average to the round's common base. The uncompressed implementation sends
full local models and averages them. These are algebraically equivalent without
compression in exact arithmetic, but floating-point operation order can differ;
bitwise equality with the uncompressed baseline is not required.

BatchNorm affine parameters train. Its 15 running-statistic/counter buffers
remain equal to the initial server values, including dtype. Clients verify them,
compressed uploads omit them, and the server restores them after aggregation.
Uncompressed uploads and all full-state downloads include them. Therefore the
QSGD-versus-uncompressed byte comparison includes this small buffer-omission
difference. QSGD-versus-LLZ uses the same omitted-buffer policy and isolates the
second stage more directly.

Source locations: [task](../../flower_face/finetune_task.py),
[client](../../flower_face/client_app.py), [server](../../flower_face/server_app.py),
[adapter](../../federated_compression/updates.py), and
[codec specification](../../compression/README.md).

The quantization RNG is derived from seed, round, and partition ID, separately
from training/data-loader randomness, with sorted tensor traversal. For an
identical source update, QSGD and LLZ operate on the same quantization draw.
After p>0 changes the model, subsequent updates and quantized symbols can
differ even with matching RNG seeds. Do not describe those trajectories as
sharing identical input gradients. Exact p=0 replay is checked separately.

## 4. Completed evidence

Current study:
[`outputs/finetune-compression-seed42/compression-study-20260925T091315807360Z/`](../../outputs/finetune-compression-seed42/compression-study-20260925T091315807360Z/).
Its `study.json`, `summary.json`, and per-run artifacts are the numerical source.
The table reports accuracy **at the minimum-loss checkpoint** and communication
for **all 15 rounds**, not bytes only up to that checkpoint. MB = 1,000,000 bytes.

| Method | Selected round | Validation accuracy | Validation loss | Training upload MB | Total serialized MB |
|---|---:|---:|---:|---:|---:|
| Uncompressed | 15 | 83.3% | 0.6181 | 2028.759 | 6086.285 |
| QSGD | 12 | 83.4% | 0.6216 | 506.835 | 4564.641 |
| QSGD + LLZ p=0 | 12 | 83.4% | 0.6216 | 59.749 | 4117.555 |
| QSGD + LLZ p=1 | 14 | 78.7% | 0.7840 | 12.483 | 4070.289 |

QSGD reduced training uploads by 75.02% versus uncompressed training. LLZ p=0
then saved 88.21% of QSGD upload bytes and 9.79% of QSGD total bytes, with exact
replay across 15 rounds and the selected checkpoint. Relative to p=0, p=1 saved
another 1.15% of total bytes while losing 4.7 accuracy percentage points. This
supports p=0 as the present working setting, not universal superiority.

Prior studies provide context but are not pooled with this pilot:

| Earlier study | Verified result | Why kept separate |
|---|---|---|
| Frozen-feature linear head, seeds 42/43/44 | Uncompressed 72.47% +/- 0.46 pp sample SD; QSGD/p=0 72.40% +/- 0.40 pp; p=1 66.37% +/- 0.93 pp | Different trainable model and training schedule |
| Uncompressed final-stage fine-tuning, seeds 42/43/44 | Selected setting 83.70% +/- 0.61 pp sample SD | Historical buffer/source implementation; not a matched compressed comparison |

Sources are `outputs/confirmed-head-compression-three-seeds/summary.json` and
`outputs/finetune-layer4-batch/study.json`. The current 83.4% is a single-seed
result and must not be compared as if it were a three-seed mean. The current
0.1-point QSGD advantage represents one validation image, not evidence of an
accuracy improvement. The last affected implementation test batch reported 84
passes; no tests or training were rerun for this documentation task.

## 5. Metrics and measurement boundaries

### Learning and distortion

Report per-seed minimum-loss checkpoint round, its top-1 validation accuracy and
cross-entropy loss, and final-round accuracy/loss separately. Save curves by
round and cumulative communication. Use example-weighted validation aggregation.

For each round, sum squared errors and squared reference norms across clients
and tensors before division:

```text
QSGD relative squared error = sum ||Q(delta)-delta||^2 / sum ||delta||^2
LLZ relative squared error  = sum ||L(Q(delta))-Q(delta)||^2 / sum ||Q(delta)||^2
```

These denominators differ, so the ratios cannot be added. Record numerator,
denominator, coordinate counts, and maximum absolute symbol error. Zero-reference
rounds use the implementation's zero ratio convention and should be identified
using the saved norms, rather than interpreted as ordinary nonzero-update samples.
The existing study summary averages round ratios equally; it is not a pooled
ratio over the whole run. Hamming mismatch fractions are available in standalone
checks, but per-round federated mismatch counts would require added instrumentation.

### Communication

The authoritative definition is
[`communication.MEASUREMENT`](../../flower_face/communication.py): serialize a
Flower Message and each unique descendant object once per message; count the
objects again for each recipient, phase, and round. Include tensor packets,
headers, padding, names, configuration, metrics, and observed message metadata.

Maintain separate quantities:

```text
B_total = B_train_upload + B_train_download + B_validation_upload + B_validation_download
saving(candidate, reference) = 100 * (1 - B_candidate / B_reference)
codec ratio = complete_codec_packet_bits / raw_input_tensor_bits
```

Use the same B boundary in numerator and denominator and state the reference.
For correspondence with the revised paper, a codec-bit/raw-FP32 ratio may be
reported as an additional metric. It is not interchangeable with our total
serialized-message saving. Counts from QSD1 and QLP1 already include codec
metadata; do not add it twice or add nested packet counts onto message totals.

Our validation phase downloads the full model again to all four clients.
Thus each round includes two sets of full-state downloads. This evaluation
choice explains much of the remaining traffic and is not an unavoidable cost
of every FL deployment. Preserve evaluation frequency across methods. A future
study of less frequent or centralized validation would be a separate protocol.

Exclude transport/RPC envelopes, TCP/IP/TLS, polling, acknowledgments, runtime
control, unobserved retries, and runtime cache savings. Also exclude dataset/cache
preparation, prefix distribution, earlier warm-start training, local checkpoints,
and held-out evaluation. Failed-message observations remain in their attempt
artifacts; failed attempts are not silently included as successful trials.
Do not call these measurements captured network traffic or derive a measured
network speedup from them.

### Computation: specified now, instrumented next week

Use a monotonic high-resolution clock. Define client encoding as tensor
conversion/delta construction through quantization, LLZ, and packing; separately
time diagnostic distortion calculations so validation overhead is visible.
Define server decoding as unpacking/dequantization and conversion to state
tensors, excluding aggregation. Define round wall time from configuration of
training messages through completion of validation, and report training versus
validation phases separately where possible.

Record per-client, per-round timings; summarize median and spread, and show
the slowest participating client when discussing synchronous latency. Do not
sum parallel client wall times and call that elapsed round duration. Record
CPU/GPU, thread counts, concurrency, software versions, and warm-up policy.
Existing run elapsed times include more work and do not substitute for these
codec measurements. No isolated codec-speed conclusion is drawn from this pilot.

## 6. Bounded sequence of future experiments

These are planned studies, not authorization to launch jobs from this document.
Finish each agreed batch, audit once, and decide before expanding it.

| Stage | Exact scope or decision rule | Gate and status |
|---|---|---|
| A: Current reference | Seed 42; none, QSGD, p=0, p=1; 15 rounds; levels 127; window 128 | Completed; preserve artifacts |
| B: Seed confirmation | Seeds 43 and 44; none, QSGD, p=0 only; six additional 15-round runs | Planned after paper audit; reuse seed 42 only if source, initialization, and protocol compatibility are verified |
| C: Focused lossy study | One controlled change, preferably feedback on/off, with matched QSGD and LLZ variants | Design proposal; implementation and exact batch require a new decision |
| D: Sparsity comparator | Top-k with value/index/header accounting, calibrated to measured upload budget | Planned comparator, not presently implemented; feedback policy must be matched or shown as a separate factor |
| E: Heterogeneity | Same global image split; redistribute training only with recorded client-class counts | Deferred until reference and partition implementation are ready |

Any training/codec correction found by the audit creates a new versioned study
family with fresh matched controls. Do not silently mix corrected runs with the
old source snapshot. Instrumentation may also change message size when metrics
are transmitted: count those bytes and either match instrumentation across
methods or report it separately. Reusing the existing seed-42 result requires
its frozen application and compatible measurement definition, not just matching
hyperparameter names.

For Stage C, a concrete candidate is seed 42, levels 127, window 128, p=1,
with feedback applied to the complete reconstructed QSGD+LLZ delta. Pair it
with QSGD using the same feedback policy. Persist one residual per client/tensor
and specify restart behavior. Residual-free and feedback-equipped results must
be separate methods. This is motivated by the paper and literature, not evidence
that feedback will recover the missing accuracy. Do not simultaneously tune
the model, training duration, p, levels, and window.

For Stage D, calibrate k using training updates and complete packet bytes without
test labels, then freeze it. Compare measured rates rather than assuming a
retained-coordinate fraction equals a byte fraction. Report the achieved budget
mismatch; do not claim a budget-matched comparison if it was not achieved.

For Stage E, the proposed first contrast is IID versus Dirichlet label skew at
alpha=0.3, four clients, fixed partition seed, and the same global train/validation/
test images. Save the allocation algorithm and its resulting counts; ordinary
Dirichlet allocation need not be sample-balanced and should not be described as
the revision's balanced protocol without enforcing that constraint. Keep global
validation evaluation fixed. Require nonempty training clients using an explicit,
recorded resampling rule. Stronger skew and additional datasets follow only after
this contrast is understood. No non-IID capability or result is claimed here.

## 7. Analysis and decision rules

For Stage B report all three seed values, arithmetic mean, sample SD (`n-1`
denominator), and paired per-seed accuracy/loss and communication differences.
Round-to-round fluctuations are not independent replicates. Three seeds give
limited uncertainty information, especially because they reuse a selected warm
start and the same validation set. Do not equate a nonsignificant result with
equivalence or report SD for the single-seed pilot.

Correctness gates are strict: at p=0 require exact symbols and reconstructed
float bytes in standalone checks, zero added LLZ error, and matching per-round
model hashes, learning metrics, and selected checkpoint in matched runs. At
p>0 require `max_symbol_error <= p`, finite decoded updates, valid packet
round-trips, and synchronized dictionaries. Retain edge-case/malformed-packet
checks and the paper's worked example; run affected tests after relevant edits.

A proposed practical screening margin for future lossy work is at most 1.0
percentage point mean validation-accuracy loss versus its matched QSGD method,
alongside an actual communication benefit. This is an engineering decision aid
for supervisor discussion, chosen after the pilot, not a formal non-inferiority
test. Present the complete trade-off even when a method misses the margin.

If a future time/bytes-to-target analysis is needed, specify the target and
persistence criterion before that batch. A possible continuation endpoint is
80% validation accuracy in two consecutive rounds; this is motivated by the
observed pilot and must be labeled exploratory if applied retrospectively.
Methods that never meet it are reported as not reached within the budget, not
assigned an invented convergence time.

Select final configurations using validation only. Once choices are frozen,
evaluate the reserved test set under a declared final protocol and report all
chosen methods. Do not feed those test results back into selection or turn this
closed-set split into an unseen-identity claim.

## 8. Reproducibility and provenance

Current application source SHA256:
`e6317d6097ad8136546eb76d97a977844534c5a2ca705e99c1af1bf0657096c9`.

Current feature-manifest SHA256:
`ae734e6a2d0e099bdc7169c934d480e17a4db90d588bad735e09a3929bea9e0a`.

Initial model-state SHA256:
`dbc0b11e65e36995957ccf6a48390211e4dc08e1b52a1784d208647bdac520b9`.

Study summary SHA256:
`be83896b7e906bf08121acf7c430c450198932ad07dfb66b6f76688dedf83492`.

Recorded environment: Python 3.11.7, Flower 1.36.0, PyTorch 2.6.0,
torchvision 0.21.0, NumPy 2.4.6, Ray 2.55.1. Deterministic algorithms enabled;
one PyTorch thread; fixed partition aggregation order. Exact replay is scoped
to the recorded data, source, software, and hardware, not promised across platforms.

Keep configuration, source snapshot, manifest and cache/weight fingerprints,
initialization, history, checkpoints, communication ledger, failure attempts,
and summary together. Preserve the saved source snapshot even if only reporting
code changes later. Raw data and outputs remain ignored; keep the protocol,
literature synthesis, source code, and experiment templates in Git.

For a new execution of the reference configuration (not an exact-source replay
unless the recorded snapshot is used):

```powershell
.venv\Scripts\python.exe -m flower_face.compression_study --config experiments\digiface-finetune-compression.toml --output-dir outputs\finetune-compression-seed42
```

It creates a new timestamped study. Use its printed directory with
`python -m flower_face.report` after completion. Follow the existing README's
local-port workaround if this Windows machine still reserves the default ports.
Do not run this merely to regenerate a report for completed results.

## 9. Monday milestone versus later implementation work

The literature synthesis, research questions, experimental controls, metrics,
evidence table, limitations, and staged design are now written. The remaining
supervisor decisions are listed in [the preparation index](README.md).
Full paper-to-code certification, timing instrumentation, feedback, Top-k,
non-IID support, and additional training are explicitly future work. Their
absence does not prevent completion of these preparation documents, but they
must not be represented as completed implementation features.
