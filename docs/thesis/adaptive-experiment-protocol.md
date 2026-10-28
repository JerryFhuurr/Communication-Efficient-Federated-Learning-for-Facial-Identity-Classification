# Frozen adaptive-LLZ pilot protocol

Prepared 21 October 2026, bringing forward the tasks planned for 23 and 26 October.
The machine-readable `protocol.json` in the study output is the authoritative
frozen instance, including the selected budgets, timestamp and source hash.

## Calibration before training

Reuse the completed, audited training-update collection
`outputs/ae-calibration-20261005T122126176771Z`. Use exactly two full deltas per
client from its reconstruction-validation trajectory: snapshot 24 (the middle
saved snapshot) and 47 (the last), for eight full client updates. The trajectory
name does not mean facial identity validation: these updates were computed
solely from training images. The source collection contains independent reset
local trajectories, not later global-model FL rounds.

Verify manifest, initialization, training configuration, tensor shapes, fixed
buffer names, source file SHA-256 and delta SHA-256. Reuse the existing feature
cache and pretrained weights. No new training is needed for calibration.

For each update, use seed 42, client-specific codec RNG and codec round 1. Draw
QSGD once per tensor, levels 127. Encode/decode LLZ p=0/1/2 with window 128 and
measure each candidate's additional squared error divided by the original
update's squared norm. Cost complete adaptive replies, including diagnostics,
with surrogate routing IDs. These are local preflight costs, not transmitted
network traffic; actual runtime messages are recorded separately.

Freeze the following deterministic rule before viewing new validation results:

1. Tight budget: lower quartile of the eight p=1 relative squared errors,
   rounded **down** to two significant digits. If this quartile is zero, use
   half the smallest strictly positive candidate error before rounding.
2. Loose budget: the larger of the upper quartile of the eight p=2 errors and
   1.1 times the tight budget, rounded **up** to two significant digits.
3. Require two distinct finite positive limits; stop if the saved updates cannot
   identify positive errors. Do not silently substitute performance-tuned values.

This is a pragmatic calibration rule to test two operating points. Eight
early local-trajectory deltas do not establish that the selected budgets are
optimal or represent late federated rounds. No validation-accuracy threshold
or literature-based accuracy guarantee is assigned to either budget.

## Exactly five training runs

| Method | Upload policy | Error feedback |
|---|---|---|
| QSGD | levels=127 | None |
| QSGD+LLZ p=0 | levels=127, window=128, exact secondary coding | None |
| QSGD+LLZ p=1 | levels=127, window=128, fixed lossy coding | None |
| Adaptive tight | Same QSGD draw; p=0/1/2; frozen tighter budget | None |
| Adaptive loose | Same QSGD draw; p=0/1/2; frozen looser budget | None |

All use one frozen app source, seed 42, four fully participating IID clients,
15 rounds, one local epoch, batch size 16, SGD learning rate 0.003 and zero
weight decay. Keep the existing 100-identity DigiFace manifest and 224px
ResNet-18 layer3 feature cache with original/flip views. Train the same layer4
and classification head; preserve the 15 fixed BN running-stat/counter buffers.
Start from the same verified initial state. Downloads remain uncompressed.

Training and augmentation RNG and partition-order aggregation retain their
existing rules. Select the best checkpoint by lowest validation loss, earliest
tie wins; also retain the final checkpoint. Report all-15-round communication
even if a selected checkpoint comes earlier. Do not evaluate held-out test data.

The sequential order is QSGD, fixed p=0, fixed p=1, adaptive tight, adaptive
loose. Elapsed launch times include packaging/startup, training, coding,
validation and checkpoint writing; do not interpret them as isolated codec
benchmarks. Adaptive client selection and server decode timings are separately
logged. Historical results provide context; they are not substituted as
source-matched controls after implementation changes.

## Selector fallback and audit

For each client update, choose the minimum-byte feasible candidate, lower p on
ties. The constraint applies to additional LLZ error relative to decoded QSGD,
normalized by the original local update's squared norm. A zero-norm update
requires zero extra error. p=0 is always feasible. A codec or numerical failure
stops the run rather than relaxing the budget.

The runtime gate already passed on the existing synthetic task. For this full
pilot, require complete rounds/client participation and reconciled serialized
communication ledgers/checkpoints. Fixed p=0 must replay all QSGD model hashes,
learning metrics and selected checkpoint exactly. For each adaptive run require
60 client-round diagnostic rows, feasible minimum-byte selections, packet-size
agreement, and matching preflight/server-observed serialized sizes.

All results, snapshots, logs and reports stay under ignored `outputs/`. Resume
reuses valid completed methods from the same frozen batch, with no added seeds
or extra experiments. Optimizers, non-IID partitions, new datasets, new AE
training, additional budgets and the later presentation update are outside
this batch.

## Interpretation and theoretical boundary

The questions are whether a measured extra-error limit retains more accuracy
than fixed p=1 while reducing uploads relative to p=0, how the selected p
changes by client/round, and what local selection overhead it incurs. A fixed
budget can yield a changing p; this is not a learned or time-varying budget
schedule. The pilot has one seed and does not establish statistical superiority.

Use the [literature assumption map and theory outline](adaptive-llz-theory-outline.md)
alongside the results. A measured reconstruction constraint is not a validation
accuracy constraint or a complete convergence theorem. The reported bytes
include packet, metadata and diagnostic costs at the logical serialized Flower
object boundary; physical network traffic is not measured.

## Completed instance

The frozen pilot completed on 21 October; see the
[local report and plots](../../outputs/adaptive-study-20261021T104244874038Z/report.md).
All five methods finished 15 rounds, p=0 passed exact replay, and the adaptive
budgets/byte-selection audit passed for 60 client-round records per policy.
The two calibrated budgets were retained unchanged. No additional seeds or
training conditions were added. The incomplete first QSGD attempt and its
successful recovery are retained in the output's runtime notes.
