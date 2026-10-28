# Adaptive LLZ implementation contract

Implemented 20 October 2026, completing the implementation work planned for
21–22 October. This document fixes the selector definition; the Friday budget
calibration and subsequent DigiFace accuracy comparison remain separate tasks.

## Update and error definitions

For each client and communication round, take the non-fixed, floating model
delta `u`. Iterate over tensor names in ascending order. Draw QSGD **once per
tensor**, with the existing stream
`SeedSequence([seed, round, partition_id, 0x51534744])`. Let `z` be its decoded
reconstruction. Every LLZ candidate uses those same integer codes and norms.

Try `p in {0,1,2}`, with the configured dictionary window. Each candidate uses
one p across the whole client update; packets remain independently framed per
tensor. Decode each actual packet locally and measure in float64, after its
float32/64 output conversion:

```
N        = sum_t ||u_t||²
E_QSGD   = sum_t ||z_t - u_t||²
E_LLZ(p) = sum_t ||z_p,t - z_t||²
E_total(p) = sum_t ||z_p,t - u_t||²
```

The feasible set has `E_LLZ(p) / N <= llz-budget`. This is a **relative squared
error**, not a relative L2 error: budget 0.01 permits an extra L2 error up to
10% of the original update norm. It does not constrain total compression error
or validation accuracy. `E_total` is measured directly; QSGD and LLZ squared
errors cannot simply be added because of the cross term.

When N=0, require E_LLZ=0; introduce no epsilon. Squared-norm overflow or
underflow that makes a nonzero vector appear to have zero squared norm/error
is rejected. Inputs must be finite float32/64 tensors; an update must have at
least one coordinate. A zero budget must preserve the QSGD reconstruction.
It need not choose p=0: p>0 can occasionally produce exactly the same values.

`p=0` is always a feasible lossless fallback. A malformed packet, invalid
configuration or codec failure raises an error rather than silently raising
the budget or switching algorithms. The original `llz-p` setting belongs to
the fixed-p method and is ignored by the adaptive method.

## Objective, ties and bytes

Choose the feasible candidate with the fewest **serialized reply object graph
bytes**; break exact byte ties by lower p. Increasing p is not assumed to
reduce size monotonically. The standalone NumPy selector accepts a caller's
cost callback; without one its explicit default boundary is the sum of packet
bytes, including packet headers/padding, excluding any external bundle framing.

The Flower adapter costs complete candidate replies, including packet names,
codec/policy metadata, all candidate diagnostics, metrics, client identity and
message framing. All candidates have the same routing context. Candidate
costs and timing are transmitted as fixed-width floating-point diagnostics so
filling these fields does not change the preflight message size. Only the
selected packet set is uploaded; unselected packets stay local. The adapter
asserts that the final reply has the size used in selection.

The existing server ledger remains authoritative for messages actually
observed at the Grid boundary. Preflight and observed bytes are stored
separately; runtime check reconciliation requires equality in that environment.
Neither quantity measures network traffic, transport envelopes, retries or
TLS/TCP/IP headers. Diagnostics are included in measured communication, so
small synthetic messages can be larger than QSGD due to this overhead.

## Software and saved diagnostics

| Component | Responsibility |
|---|---|
| `compression/adaptive_llz.py` | Pure NumPy quantization, candidate coding/decoding, distortion measurements and deterministic selection |
| `federated_compression/adaptive_llz.py` | Torch deltas, Flower framing, complete-reply costing and validation of the selected upload |
| `flower_face/client_app.py` | Runs local training, preserves fixed buffers, then invokes adaptive encoding |
| `flower_face/server_app.py` | Validates the dynamic p, decodes deltas, applies the existing sample-weighted aggregation and records diagnostics |
| `flower_face/check_adaptive_llz.py` | Bounded three-run, two-round, four-client runtime/replay gate on the existing synthetic manifest |

Each adaptive experiment saves `adaptive_selection.jsonl`, one row per client
per round, with:

- candidate p, packet bytes, full reply bytes, extra LLZ squared error and total
  squared reconstruction error;
- selected p, configured budget, original-update norm, QSGD error, selected LLZ
  code/float errors and counts;
- client selection seconds, server decode seconds, and server-observed reply
  bytes.

Client selection timing covers delta materialization, QSGD, all candidates'
encoding/decoding, diagnostics and preflight costing. It ends before the final
reply construction and its size assertion. Server decode timing excludes
aggregation and logging. These are CPU simulation timings, not network latency.
Candidate costs are preflight sizes, not three actual uploads.

Round history adds selected-p client counts, summed selection time, and pooled
extra error divided by pooled original-update norm. The existing
`llz_relative_squared_error` still uses decoded-QSGD norm as its denominator;
it must not be confused with the adaptive budget's original-update denominator.
Per-client logs are needed to check individual constraints; a pooled ratio
alone would be insufficient.

The server validates exact method/round/level/window/budget metadata, selected-p
membership, finite candidate diagnostics, reported feasibility and optimality,
payload byte counts and packet shapes/types. It cannot independently recompute
error against the client's unavailable uncompressed update. Honest client
measurement is an explicit assumption. No adversarial-client guarantee or
cross-platform bitwise replay guarantee is claimed.

## Use and correctness boundary

Run the bounded integration gate from the repository root:

```powershell
.venv\Scripts\python.exe -m flower_face.check_adaptive_llz
```

It reuses `data/synthetic/manifest.json`, freezes one source snapshot and runs
exactly QSGD, adaptive budget 0, and adaptive budget 1.0, each with seed 42,
four clients and two rounds. Budget 1.0 is deliberately permissive to exercise
a positive-p upload; it is **not** a proposed thesis accuracy setting. The
audit reconciles every message, all client/round diagnostics and checkpoints.
Zero-budget models, learning metrics and selected checkpoint must match QSGD
exactly; codec diagnostics and timings intentionally differ. Held-out test data
is unused. JSON, logs, source snapshot and checkpoints stay under ignored
`outputs/adaptive-check-*/`.

The regular launcher now accepts
`--compression qsgd-llz-adaptive --llz-budget <relative-squared-error>` alongside
the existing task/manifest/model flags. Do not treat the old fixed-p study
runner/report matrix as an adaptive study runner: the Friday protocol will
freeze two budgets and specify its matched five-method batch separately.

## Connection to the literature

Use the existing [focused review](adaptive-llz-literature-review.md) and
[meeting literature brief](adaptive-llz-meeting-brief.md): QSGD fixes the first
stage, the [LLZ paper audit](llz-paper-audit.md) defines the symbol codec, and
AdaQuantFL/DAdaQuant motivate adaptation. This implementation adapts the second
stage after a fixed QSGD draw. Positive-p selection can be biased; the
unbiased-compressor assumptions in the reviewed quantization analyses have not
been established for it. A reconstruction constraint and exact zero-budget
replay are implementation properties, not a new convergence theorem.
