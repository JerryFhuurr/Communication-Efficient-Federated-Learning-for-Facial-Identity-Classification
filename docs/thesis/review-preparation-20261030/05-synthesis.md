# Block 5: Cross-method synthesis and citation audit

Prepared and checked **29 October 2026**; Friday work brought forward.
Status: **complete for the five-block preparation scope**, with the source
and scientific limitations below retained. This is the writing handoff for
the 4 November literature-review PDF, not the finished PDF or a new experiment.

Read [progress](progress.md) and the [review outline](../literature-review-outline-2026-11.md)
for scope and deadlines. References below use keys in [references.bib](../references.bib);
source locations are in Blocks 1-4. These tables compare mechanisms and evidence,
not numerical accuracy across unrelated papers.

## 1. Integrated algorithm comparison: representations and guarantees

| Method / reference key | Representation / original setting | Error assumptions and feedback | Practical costs / thesis relevance |
|---|---|---|---|
| QSGD (2017), `alistarh2017qsgd` | Norm plus stochastic signed levels; distributed SGD | Unbiased numerical quantizer with relative variance bound; EF is a separate choice | Norms, shape and the actual source code matter. Our dense bitstream differs from the original entropy-style encoding. |
| signSGD (2018), `bernstein2018signsgd` | Coordinate signs and majority-vote aggregation | Conditional sign-optimization theory; additional noise assumptions for improved worker scaling | Sign/tie convention and framing; changes optimization and aggregation, so not a FedAvg codec swap. |
| Top-k / Rand-k, `stich2018memory` | Selected values and coordinate set | Top-k and unscaled Rand-k contract but are biased; scaled Rand-k is unbiased with increased variance | Count indices or shared-mask setup. Supervisor's L-Set denotes this coordinate-selection family. |
| Memory / EF (2018-19), `stich2018memory`, `karimireddy2019error` | Residual-corrected future inputs, not its own wire format | Results require the stated compressor and optimization conditions | Dense persistent client state; our delta-space recurrence needs separate local-epoch analysis. |
| DGC (ICLR 2018), `lin2018dgc` | Sparse accumulated updates plus momentum-related mechanisms | Empirical system includes clipping, masking and warm-up | Value/run-length coding and dense state; current Top-k+EF is not an implementation of DGC. |
| PowerSGD (2019), `vogels2019powersgd` | Low-rank factors via dependent collective operations | Biased approximation, residuals and warm starts | Matrix products, orthogonalization and topology matter; arbitrary independent factor uploads do not reproduce it. |
| FedPAQ (2020), `reisizadeh2020fedpaq` | Quantized model differences after local steps | Unbiased relative-variance quantizer, IID data and prescribed optimization conditions | Closer local-update precedent; positive-p adaptive LLZ has not met its hypotheses. |
| Chandar AE (2021), `chandar2021autoencoder` | Learned weight representation with client encoder/server decoder | Reconstruction/classifier feasibility; no general convergence result transferred here | Calibration, fitting, decoder distribution and drift. Our AE remains offline feasibility. |
| LLZ-p / LLZ-SI (2023/26), `zhong2023llz`, `cao2026llz` | Dictionary coding of quantized gradient symbols; optional historical side information | p=0 preserves first-stage reconstruction; positive p adds possible bias. Journal theory has conditional distortion/SGD assumptions | Pointers, literals, norms and history; LLZ-SI is not implemented in our project. |
| NNCodec FL poster (2025), `becking2025tinyfl` | Differential update reduction, quantization and arithmetic coding | Task-specific evidence; not a new general FL guarantee | Useful representation/state comparison; neural-network coding does not necessarily mean an AE latent codec. |

Definitions, costs and checked passages: [Block 1](01-quantization.md),
[Block 2](02-sparse-feedback-lowrank.md), [Block 3](03-learned-adaptive.md),
[Block 4](04-recent-work-search.md), and the
[IEEE comparison](../llz-ieee-version-comparison-2026-10-29.md).

## 2. What adaptation changes

| Method / key | Adapted object and objective | State / evidence boundary | Implication for our LLZ question |
|---|---|---|---|
| AdaQuantFL, `jhunjhunwala2021adaquantfl` | Quantization precision over training, motivated by a learning bound under a bit budget | Loss-based approximation and initial precision choice; idealized bit cost | A precedent for adapting first-stage precision, not a universal tau. |
| DAdaQuant, `honig2022dadaquant` | Time-dependent levels and client-weight allocation | Loss history; variance-model allocation with a sum-of-levels proxy | Adaptation across rounds/clients already exists; equal weights remove weight-driven allocation differences. |
| KLMS, `isik2024sideinformation` | Common-randomness sample coding and divergence-sensitive blocks | Shared distributions/seeds, boundaries and refresh; conditional distributional approximation | Side information differs from reconstructed-symbol dictionaries and exact replay of a fixed QSGD draw. |
| ComParFreFL, `yan2026robustcompression` | Optimizer robust to compression without compression-dependent step sizes | Momentum/control-variate/EF recursion; expected relative error strictly below one | Its optimizer and assumptions are not established for our no-EF FedAvg selector. |
| ASCEND, `liu2026ascend` | Per-client candidate strategy using reward-based bandit selection | Public distillation data, residuals and rollback; preprint with unresolved proof-report link | Candidate selection is existing work. Our constrained same-draw byte minimization is a narrower distinction. |
| EA-AQF, `ikram2026eaaqf`; PAC-MCoFL, `tan2025pacmcofl` | Resource-dependent training/precision or joint assignment/quantization policies | Metadata/abstract screening only; changes more than the codec | Contextual alternatives, not source-matched fixed-model controls. |
| Our adaptive LLZ | Choose one p in {0,1,2} after fixed QSGD; smallest complete feasible reply | Locally decode all candidates; additional error divided by original squared norm; no EF | Working implementation; budget rationale and learning preservation remain open. |

R3 discrepancy-aware calibration remains a close **abstract-only, access-limited**
alternative in Block 4. It is intentionally not presented as a fully verified
formal reference or a fully inspected algorithm in these tables.

## 3. Cross-method conclusions and qualified research gap

The literature supports treating precision reduction, coordinate selection,
low-rank approximation, learned representation and source coding as composable
operations. EF is a state mechanism that can wrap a compressor; adaptation
selects settings or policies. Neither is a mutually exclusive representation
family. Message size, repeated local computation and optimization quality must
be considered together.

**Proposed review gap paragraph:** Existing research already adapts quantization
across training and clients, exploits shared side information and selects
compression strategies using learning feedback. LLZ provides a further coding
stage for quantized updates, while its journal article identifies adaptive
choices as future work. This thesis therefore investigates a narrow question:
whether a finite LLZ tolerance selector, evaluated after one fixed QSGD draw,
can reduce complete upload bytes under a declared additional-error constraint,
and how to set that constraint to retain an independently evaluated learning
criterion. The bounded recent search motivates this question but establishes
neither priority nor a new convergence theorem. The inaccessible calibration
paper and the assumptions of close alternatives remain explicit limitations.

For clarity, u denotes the local model delta, z decoded QSGD and z_p decoded
LLZ. Our constraint is `sum ||z_p-z||^2 <= tau * sum ||u||^2`; it is not a bound
on total compression error or loss increase. Integer p, QSGD levels, relative
squared-error budget tau and paper retained-bit fraction rho have different
meanings. p=0 exact replay is an implementation check relative to QSGD. Neither
it nor a small validation difference proves statistical equivalence or FL
convergence. Historical tau=5.9/7 values remain calibration heuristics.

## 4. Separate project evidence: frozen pilot, not literature ranking

The [saved summary](../../../outputs/adaptive-study-20261021T104244874038Z/summary.json),
[protocol](../../../outputs/adaptive-study-20261021T104244874038Z/protocol.json) and
[report](../../../outputs/adaptive-study-20261021T104244874038Z/report.md) record
seed 42, four IID clients, 100-way DigiFace classification, 15 rounds and one
local epoch. Cached ResNet-18 layer-3 features feed trainable layer4/classifier.
No EF; checkpoint selected by minimum validation loss; held-out test unused.
These are historical Windows runs, not new WSL measurements.

| Method | Selected validation accuracy | Training-upload MB | Total serialized MB |
|---|---:|---:|---:|
| qsgd | 83.40% | 506.836 | 4564.641 |
| qsgd-llz-p0 | 83.40% | 59.749 | 4117.555 |
| qsgd-llz-p1 | 78.70% | 12.483 | 4070.289 |
| adaptive-tight | 83.20% | 38.570 | 4096.375 |
| adaptive-loose | 78.50% | 13.278 | 4071.084 |

MB means 1,000,000 bytes. Upload here means **training replies**; total includes
training and validation exchanges in both directions. All bytes cover the full
15 rounds, even if a checkpoint was selected earlier. They are serialized
Flower objects including metadata/diagnostics, not physical network traffic.

Recalculated from the saved integers, adaptive-tight saves **35.45%** of
training-upload bytes versus p=0 but **0.514%** of total serialized bytes,
with **0.20 percentage points lower** selected validation accuracy. Fixed p=0's
saved all-round/checkpoint replay audit passed. The looser policy is worse than
fixed p=1 in both selected accuracy and training-upload bytes in this pilot.
These observations do not establish superiority, equivalent accuracy or a safe
budget: one seed, changed trajectories, checkpoint selection and candidate
search overhead limit interpretation. The earlier three-seed fixed-method
study is separate; do not pool it with this adaptive pilot or label Top-k+EF
as an EF-matched control.

The offline AE evidence remains in Block 3 with its own calibration/probe
scope. Do not put it into this table as if it had an FL accuracy result.

## 5. Citation audit and unresolved evidence

Existing core keys were checked against the source registers in Blocks 1-3;
recent metadata was checked against primary landing pages on 29 October.
Five entries were added: ComParFreFL, ASCEND, NNCodec FL, EA-AQF and PAC-MCoFL.
Publication/reading-depth notes remain in their entries. R3's full-text/venue
verification remains pending and no fabricated BibTeX entry was added for it.

The IEEE Early Access entry and earlier manuscript intentionally share a title
but have different keys/provenance. They are not accidental duplicate papers.
Use the published entry for new journal claims, keeping the old key where it
records the source actually used by an earlier audit. Unread or abstract-only
work must not be promoted to full-method evidence when writing the PDF.

| Unresolved item | Effect on the review | Required handling |
|---|---|---|
| R3 live OpenReview verification | Full method/results and live venue status unverified | Keep abstract-only flag; later seek legitimate author/supervisor copy. |
| ASCEND v1 proof-report reference | Stated assumptions/results inspected, proof details unavailable there | Attribute the claims; do not use them as a validated proof for our method. |
| Recent-search scope | Seven candidates, three new detailed readings; not exhaustive | Retain qualified gap language and actual 29 October cutoff. |
| Budget justification and local-epoch theory | No learning-safe tau or general convergence proof established | Separate empirical acceptance from codec correctness; freeze the next protocol independently. |
| Numerical evidence | Adaptive pilot one seed; AE offline; logical byte boundary | Label protocols and avoid cross-paper or mixed-feedback leaderboards. |
| PDF release | Source notes and metadata audit are complete; final layout is not | Draft, render and inspect all figures/tables/citations before sharing. |

Checks cover citation-key resolution, balanced BibTeX fields and duplicate
identifiers, local links, source fingerprints, table arithmetic, scope labels
and Git whitespace. This is an editorial/source audit, not a proof audit or
experimental reproduction. An ignored machine-readable record is saved as
`outputs/literature-blocks-20261029/verification-block-5.json`.

## 6. Original figures and Monday drafting handoff

| Planned figure/table | Content and provenance | Guardrail |
|---|---|---|
| Original taxonomy | Quantization, sparsity, low rank, learned coding and source coding; EF/adaptation as overlays | Cite Blocks 1-4 sources; do not present overlays as disjoint compressors. |
| Original adaptive pipeline | Local delta -> one QSGD draw -> p candidates -> local reconstruction -> constraint/complete-byte choice | Label u, z, z_p and the exact error denominator. |
| Cost illustration | Dense, coordinate/index, low-rank and learned/setup content costs from Blocks 1-3 | Mark illustrative arithmetic; not measured network speed. |
| Algorithm and assumptions tables | Sections 1-2, split for readable PDF columns | Preserve version, assumptions, state and evidence depth. |
| Separate pilot graph | Selected accuracy versus training-upload bytes, plus total-byte panel from saved summary | Five pilot methods only, no invented error bars; historical Windows, seed 42. |

**Monday 2 November:** draft motivation and local-update/measurement definitions;
then synthesize representation families, EF and adaptive/source-coding approaches;
write the qualified gap and separate project-evidence section; integrate the
source keys and flagged limitations. Use the existing outline's space guide
without adding papers to fill pages. Every plotted number must map to a saved
source row or explicit illustrative formula.

**Tuesday 3 November:** create original diagrams and tables, compile the PDF,
resolve bibliography citations and inspect every rendered page for clipping,
small text, equations and caption/denominator mistakes. **Wednesday 4 November:**
final editorial check and user sharing. No PDF was created or released by this
preparation block. Budget experiments and the five-minute 10 November meeting
pack remain subsequent work.
