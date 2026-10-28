# Results, discussion, and limitations

Current-stage results chapter, 25 September 2026. All numbers below come from
completed local studies; no training was performed to write this chapter.
The matched fine-tuning pilot is the primary evidence. Earlier studies provide
context only. The [experimental design](experimental-design.md) identifies
source artifacts, hashes, selection rules, and planned follow-up work.

## Matched fine-tuning pilot

Four methods used seed 42, the same fixed data split and warm start, and 15
rounds with four clients. QSGD used 127 intervals and LLZ used window 128.
The table reports validation accuracy at the minimum-validation-loss checkpoint.
Upload and total bytes cover the complete run, not only the selected round.
MB denotes 1,000,000 bytes; communication is logical serialized message size.

| Method | Selected round | Selected accuracy | Selected loss | Final accuracy | Training upload MB | Total MB |
|---|---:|---:|---:|---:|---:|---:|
| Uncompressed | 15 | 83.3% | 0.6181 | 83.3% | 2028.759 | 6086.285 |
| QSGD | 12 | 83.4% | 0.6216 | 83.4% | 506.835 | 4564.641 |
| QSGD + LLZ p=0 | 12 | 83.4% | 0.6216 | 83.4% | 59.749 | 4117.555 |
| QSGD + LLZ p=1 | 14 | 78.7% | 0.7840 | 78.0% | 12.483 | 4070.289 |

The primary evidence is
[the saved summary](../../outputs/finetune-compression-seed42/compression-study-20260925T091315807360Z/summary.json).
The [accuracy curve](../../outputs/finetune-compression-seed42/compression-study-20260925T091315807360Z/report/accuracy.png)
shows the complete trajectory, and the
[communication chart](../../outputs/finetune-compression-seed42/compression-study-20260925T091315807360Z/report/communication.png)
separates upload from total bytes. These local links require the ignored result
artifacts and will not resolve from the Git repository alone.

## Interpretation by research question

**RQ1: lossless secondary compression.** QSGD and LLZ p=0 produced identical
per-round model hashes, learning metrics, and selected-checkpoint results for
all 15 rounds. LLZ reduced QSGD training uploads by 88.21%, from 506.835 MB to
59.749 MB, while total bytes fell by 9.79%. This is direct evidence of correct
lossless integration for the recorded run. It preserves the quantized trajectory,
not the original full-precision updates. Exact replay is expected from a correct
lossless implementation and is a correctness result, not a novel algorithmic
discovery.

QSGD alone reduced training uploads by 75.02% and total bytes by 25.00% relative
to the uncompressed control. Its selected accuracy was 0.1 percentage point
higher, but its selected loss was slightly higher as well. The accuracy difference
corresponds to one validation image; it does not establish improved accuracy or
statistical equivalence. The controlled QSGD-versus-p=0 comparison provides the
clearest evidence for LLZ's incremental contribution.

**RQ2: positive tolerance.** With p=1, selected accuracy fell by 4.7 percentage
points versus QSGD/p=0. Training uploads fell another 79.11% relative to p=0,
but the additional total saving was only 1.15%, or 47.266 MB over 15 rounds.
The configuration is therefore less attractive when retaining accuracy and
reducing total message volume are the immediate priorities. An upload-constrained
deployment might value that trade-off differently, but deployment conditions
were not measured here.

This result does not show that p=1 is intrinsically unsuitable. Training duration,
quantization scale, tensor grouping, feedback, and task can affect the result.
We did not isolate those factors. In particular, the supplied revision's main
experiments generally use error feedback, while this pilot does not. Its
stochastic-gradient protocol and communication denominator also differ. The
literature review and audit identify these differences; the observed accuracy
loss is not a direct contradiction of the paper.

| Method | Mean round QSGD relative squared error | Mean round added LLZ relative squared error |
|---|---:|---:|
| QSGD | 2.407885 | Not applicable |
| QSGD + LLZ p=0 | 2.407885 | 0 |
| QSGD + LLZ p=1 | 2.740243 | 1.732759 |

These are reconstruction-error ratios, not classification error rates or
percentages of incorrect identities. Values above one are possible. LLZ's
denominator is the reconstructed QSGD update; QSGD's is the original update.
The ratios cannot be added to estimate combined error. The p=1 trajectory
changes later local updates, so its QSGD distortion need not equal the other
runs' distortion despite using the same quantizer settings and RNG scheme.

**RQ3: total communication and computation.** After p=0, most remaining bytes
come from uncompressed full-model downloads. The implementation downloads the
model for training and again for validation each round. This is a property of
the chosen evaluation protocol, not an intrinsic requirement of all FL systems.
Packet savings should therefore be presented alongside full logical-message
totals. Physical network bytes, bandwidth effects, and isolated codec latency
were not measured; no speedup or energy-saving conclusion follows from this table.

**RQ4: robustness.** The stronger-model compression study contains one seed and
one approximately IID partition. Robustness across training seeds and non-IID
partitions remains unanswered. Earlier head-only evidence cannot fill that gap
because the trainable representation and training schedule differ.

## Historical context without pooling incompatible studies

The frozen-feature linear-head study used three seeds. Mean selected validation
accuracy was 72.47% with sample SD 0.46 percentage point for uncompressed training,
72.40% +/- 0.40 point for QSGD and p=0, and 66.37% +/- 0.93 point for p=1.
This is consistent with the usefulness of p=0 in that smaller setting, but is
not three-seed confirmation of the fine-tuned model.

The earlier uncompressed layer4 fine-tuning study achieved 83.70% +/- 0.61 point
across three seeds for its selected setting. It motivated using the stronger
model, but its source/buffer implementation differs from the fresh matched
pilot. These studies should remain separate tables rather than a combined mean.
Their artifact locations are recorded in the experimental protocol.

## Threats to validity and how claims are bounded

| Threat | Consequence for interpretation | Current control or remaining action |
|---|---|---|
| One seed in the matched stronger-model pilot | Between-seed uncertainty is unknown | Report individual values; paired seeds 43/44 are planned, not completed |
| Reused validation set and selected warm head | Selection can make reported performance optimistic | Disclose the selection history; reserve test evaluation until decisions are frozen |
| Shared initialization across later seeds | Seed variation does not cover the complete training/selection pipeline | Define the repeated experiment as conditional fine-tuning from that head |
| First 100 identities and synthetic images | Limited population and domain coverage | Avoid claims about all DigiFace identities or real-world face recognition |
| Closed-set image split | No evidence for unseen identities, verification, or session robustness | Name the task precisely; design different splits for different claims |
| Near-duplicate or correlated renders | Disjoint files do not guarantee statistically independent samples | Exact duplicates are checked; residual dependence remains a limitation |
| Fifteen-round budget | No proof of convergence or long-run ranking | Report finite-budget curves and selected/final outcomes separately |
| No feedback or sparse baseline | Limited comparison with published lossy systems | Label absent features; control feedback and actual byte budgets in future comparisons |
| Paper notation and custom framing | Not a literal author-code reproduction | Preserve the reconstructed-history interpretation and packet-format specification |
| Logical object accounting | Does not capture transport overhead or caching effects | State the boundary; report upload, download, and validation separately |
| Timing instrumentation absent | Smaller packets do not establish faster training | Measure codec work and round wall time as a separate implementation task |
| Local simulation and trusted clients | No claim about heterogeneous hardware, failures, or hostile behavior | Limit conclusions to the recorded simulated environment |
| Incomplete Git-only artifact distribution | A checkout alone cannot reproduce the saved run | Preserve source snapshots, data/cache/weight hashes, and relocation instructions |

## Current conclusion and next decisions

The project has a functioning, independently testable QSGD+LLZ-p pipeline and
an auditable federated integration. Under the present setting, p=0 provides
substantial incremental upload reduction with no added learning distortion.
Positive tolerance yields smaller packets but a material accuracy cost, with
a much smaller additional reduction in total communication.

Retain p=0 as the working reference. The next engineering step is timing
instrumentation; the next confirmation study is the bounded paired-seed plan.
If lossy recovery is prioritized, error feedback is a motivated candidate for
a controlled ablation, not an assumed remedy. These conclusions support the
current-stage thesis narrative while leaving generality and final test-set
performance explicitly open.
