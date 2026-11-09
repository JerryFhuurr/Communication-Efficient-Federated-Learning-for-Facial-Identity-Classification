# Block 4: Recent work and qualified novelty check

Search and review date: **29 October 2026**, Europe/Paris. The planned Friday
block was brought forward; no coverage of 30 October or later is implied.
Status: **bounded search complete; one source remains access-limited**.
Seven recent candidates were screened, including the already reviewed IEEE
LLZ anchor. Three other candidates received targeted full-text inspection.
This is a narrative-review update, not an exhaustive systematic search.

## 1. Scope, queries and selection rule

Include 2025-2026 primary sources about FL update compression, adaptive
precision/error control, learned representations or LLZ. Prefer methods close
to our fixed-model, compressed-local-update question. Exclude image/data
compression without FL update transmission, privacy-only compression, and
rate-distortion terminology used only for unrelated generalization questions.
Earlier foundational sources belong in Blocks 1-3, not this recent-work count.

The cap was eight screened candidates and three new detailed inspections.
Seven were sufficient; the eighth slot was not filled merely to reach a quota.
Publication status, abstract-only screening and targeted reading are recorded
separately below. Search-result snippets were discovery aids, not proof of
method details or experimental gains.

All queries below were executed on 29 October. Domain filters are explicit;
no result-count or ranking claim is made.

| Query ID | Exact query | Domains |
|---|---|---|
| Q1 | `"federated learning" "adaptive compression" "2025" distortion` | Unrestricted discovery |
| Q2 | `"federated learning" "compression" "2026" adaptive` | Unrestricted discovery |
| Q3 | `"federated learning" "learned compression" "2025"` | Unrestricted discovery |
| Q4 | `"federated" "Lempel-Ziv" adaptive compression` | Unrestricted discovery |
| Q5 | `federated learning adaptive compression distortion 2025` | arxiv.org; openreview.net; proceedings.mlr.press |
| Q6 | `federated learning learned compression rate distortion 2026` | arxiv.org; openreview.net; proceedings.mlr.press |
| Q7 | `federated learning Lempel Ziv adaptive distortion` | arxiv.org; ieeexplore.ieee.org; shanghaitech.edu.cn |
| Q8 | `"federated learning" "2025" "adaptive quantization"` | arxiv.org; openreview.net; proceedings.mlr.press |
| Q9 | `"federated learning" "2025" "learned" "compression"` | arxiv.org; openreview.net; proceedings.mlr.press |
| Q10 | `"federated learning" "2025" "rate-distortion"` | arxiv.org; openreview.net; proceedings.mlr.press |

Primary landing pages were opened for metadata, with PDFs used for the three
selected detailed readings. Search scope and language can miss alternatives;
search-index freshness is not guaranteed. The local machine-readable record
is `outputs/literature-blocks-20261029/recent-search-log.json`.

## 2. Screened candidates and inclusion decisions

Every row was screened on 29 October. R1 reuses the earlier paper comparison;
it does not add a fourth new detailed inspection.

| ID | Source and date/status | Evidence inspected | Inclusion decision / relation to our question |
|---|---|---|---|
| R1 | Cao et al., *Federated Learning with Lossy Lempel-Ziv Gradient Compression*, [IEEE DOI](https://doi.org/10.1109/TCOMM.2026.3745158), Early Access 21 October 2026 | Existing publisher PDF and [29 October comparison](../llz-ieee-version-comparison-2026-10-29.md) reused | Include as direct LLZ anchor; fixed-p/side-information methods and stated future adaptive direction. |
| R2 | Yan, Zhong and Zhang, *Robust Federated Learning Against Adaptive Compression*, [ICML 2026 / PMLR](https://proceedings.mlr.press/v306/yan26x.html) | Publisher metadata; targeted full text | Include and inspect: relevant optimizer/feedback theory for biased compression, not an LLZ budget rule. |
| R3 | Wang et al., *Enhancing Communication Compression via Discrepancy-Aware Calibration for Federated Learning*, [indexed OpenReview PDF](https://openreview.net/pdf/ddf3e76e6309cbae1c6038cdbf7b95cbb9e5e54f.pdf); indexed header labels ICLR 2026 | Indexed abstract/header only; live retrieval returns verification challenge | Include as a close task-sensitive alternative; full method and venue status remain unverified in this audit. |
| R4 | Ikram et al., *EA-AQF: energy aware adaptive quantization and freezing in federated learning*, [Cluster Computing](https://link.springer.com/article/10.1007/s10586-026-06503-0), 25 August 2026 | Current publisher metadata and abstract | Contextual inclusion: resource-dependent activation precision/freezing plus update quantization also changes local training. Not selected for detailed inspection. |
| R5 | Liu et al., *Adaptive Heterogeneous Compression for Resource-Efficient Federated Knowledge Distillation* (ASCEND), [arXiv:2608.15660](https://arxiv.org/abs/2608.15660), v1 16 August 2026; preprint | Landing page; targeted full text | Include and inspect: per-client candidate-compressor selection is directly relevant prior work. |
| R6 | Becking et al., *Efficient Federated Learning Tiny Language Models for Mobile Network Feature Prediction*, [arXiv:2504.01947](https://arxiv.org/abs/2504.01947), v1 2 April 2025 | Both PDF pages and metadata; authors report EuCNC & 6G Summit poster acceptance | Include and inspect: practical update coding, entropy representation and state/syntax costs. |
| R7 | Tan et al., *Pareto Actor-Critic for Communication and Computation Co-Optimization in Non-Cooperative Federated Learning Services*, [arXiv:2508.16037](https://arxiv.org/abs/2508.16037), August 2025 v1/v2; preprint | Metadata and abstract | Context only; exclude from detailed comparison because it jointly controls resource allocation/client assignment, beyond our fixed-protocol codec question. |

R4's publisher page flags an updated article; its live abstract differs from
an indexed snippet. This record uses the live abstract and does not combine
numbers from different versions. Preprints and author-reported acceptance are
not silently upgraded to independently verified archival publications.

## 3. Three closest accessible works: targeted inspection

### R2: optimization robustness does not choose our budget

[Official PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/yan26x/yan26x.pdf),
PDF pp. 3-6, Algorithms 1-2, Assumptions 1-3 and Theorems 1-2.
ComParFreFL combines momentum increments and error feedback in one compressed
quantity, with normalized updates and control variates. Under smoothness,
unbiased variance-bounded gradients and a client-wise expected relative-error
factor strictly below one, its step-size prescriptions do not depend on the
compression factor. The theorem's bound still contains that factor.

**Our interpretation:** compression-independent step sizes are useful theory,
but the result analyzes a different optimizer and state recursion. Our
QSGD-plus-LLZ selector without EF has not been shown to satisfy its contraction
condition. A bound on extra LLZ error is insufficient: total error also includes
QSGD, and a conditional guarantee is needed when selection depends on history.
This reading does not establish arbitrary changing-compressor convergence for
our FedAvg implementation or tell us which tau is safe.

### R5: candidate selection is already an existing approach

[Author PDF v1](https://arxiv.org/pdf/2608.15660v1), PDF pp. 4-9, Sections III.C,
IV.B and V; Algorithm 1, Theorems 1-2. ASCEND chooses Top-K, Random-K or
Periodic-K per client using an EMA-based epsilon-greedy bandit. Its clipped
reward combines local loss improvement and server knowledge alignment per
execution time. It uses residual accumulation, heterogeneous model prototypes,
a public distillation dataset and a rollback safeguard.

Convergence statements assume smooth objectives, unbiased bounded-variance
gradients and bounded residuals. The regret analysis additionally assumes
piecewise-stationary rewards, concentration and sufficient sampling of every
strategy. These are assumptions, not consequences automatically established by
the practical exploration policy. The inspected v1 points to proofs using an
unresolved `[*]` report reference, so the proof details remain an evidence gap.

**Our interpretation:** our rule measures all candidates on the same QSGD draw
and minimizes bytes subject to a local error constraint. That differs from
learning a reward-based policy over rounds, but broad per-client candidate
selection is not our novelty. This source is a preprint, not a proof authority
for our implementation.

### R6: practical neural-network coding is not necessarily an AE

[Author PDF v1](https://arxiv.org/pdf/2504.01947v1), both pages; Section II.C,
Figure 3 and Table I. This two-page poster studies five-client, 25-round FL of
tiny language models on Berlin V2X. It codes differential updates using NNCodec:
parameter reduction, quantization and DeepCABAC arithmetic coding. A Parameter
Update Tree identifies relationships to earlier parameter versions.

**Our interpretation:** an encoder/decoder architecture and the label
neural-network coding do not by themselves mean an autoencoder latent model.
The relevant comparison is a practical coding pipeline including quantization,
entropy coding and update state. Its table provides task-specific data-size
and quality measurements; these cannot be transferred to our face task or
compared directly to total Flower serialization. A two-page poster is also
limited evidence for reproducing every setting or asserting a general
convergence guarantee. No benchmark ratio is adopted as our target.

## 4. Access gaps and claim boundaries

R3's indexed abstract describes calibrating candidate compression units using
model-output discrepancy on a small local subset, with sparsification/low-rank
applications. That suggests task sensitivity as an alternative to our update
norm proxy. The live source requires browser verification; no full-text
method, calibration cost, theorem or numerical claim was verified. Preserve
this row as **abstract-only**, and seek an accessible author manuscript or
supervisor-provided copy during later citation consolidation. No access check
was bypassed, and this gap does not justify excluding the alternative from
the novelty discussion.

For R2/R5/R6, detailed reading means checking the cited method/assumption
sections, not independently proving the results, running code or reproducing
experiments. Downloaded PDFs and hashes are in the local source manifest.
ASCEND text extraction contained an invalid Unicode surrogate; its text copy
was repaired with replacement characters, while the original PDF was retained
unchanged. The repair is not a missing-source problem.

## 5. Nearest alternatives and defensible research position

| Question | Nearest checked alternatives | Consequence for our claim |
|---|---|---|
| Has LLZ already compressed FL gradients? | R1; earlier LLZ work in the existing bibliography | Yes; implementation of LLZ alone is not the new contribution. |
| Has compression already adapted per client/round? | AdaQuantFL, DAdaQuant in [Block 3](03-learned-adaptive.md); R5 | Yes; avoid a broad first-adaptive or first-selection claim. |
| Can model/task sensitivity guide a compressor? | R3, abstract only | A close alternative exists; method-level comparison remains access-limited. |
| Can biased compression have convergence analysis? | R2; EF foundations in [Block 2](02-sparse-feedback-lowrank.md) | Yes, under specified algorithms/assumptions; we cannot inherit those results. |
| Does a low-dimensional or quantized payload give actual communication cost? | R6; Block 3 setup analysis | Representation, syntax, state and decoder/setup costs must be included. |

**Working research question, not a priority claim:** for a fixed QSGD draw and
fixed FL training protocol, can selecting LLZ p from a finite candidate set
minimize the complete transmitted reply subject to an explicitly defined
additional-error constraint, and how should that budget vary to retain a
predefined learning-quality target? The selected readings motivate this narrow
question but do not establish that it has never been studied.

A useful thesis contribution could combine a reproducible selector, transparent
byte/error accounting and evidence about budget policies. A convergence result
would require a separate derivation for the actual local-update algorithm and
compressor; empirical accuracy alone cannot supply it. The unresolved 5.9/7
heuristics are not validated by any source in this search.

## 6. Handoff to Block 5

Carry these seven rows, three inspected comparisons and R3's access limitation
into the integrated algorithm table. Block 5 consolidated verified metadata in
[references.bib](../references.bib), distinguishing preprints, poster evidence
and publisher versions. The source URLs remain available here. R2/R4/R5/R6/R7
map to `yan2026robustcompression`, `ikram2026eaaqf`, `liu2026ascend`,
`becking2025tinyfl` and `tan2025pacmcofl`; R3 remains access-limited. Do not
expand the search just to fill the eighth slot; revisit R3 when legitimate
access is available, and retain the qualified gap wording in the PDF draft.
The final PDF, illustrative figures/tables and experiments remain later work.


## 7. Dated follow-up on the closest candidate (2 November 2026)

The 29 October screen included seven candidates and completed three planned detailed readings. A follow-up on 2 November retrieved the final OpenReview PDF for R3; its first page identifies it as published at ICLR 2026. The abstract, related-work and method sections were checked. The paper scores candidate coordinates or singular components by output discrepancy on local calibration data, then selects under a communication budget. This is a close task-sensitive alternative and narrows the thesis claim to LLZ tolerance selection after one fixed QSGD draw, complete-reply byte cost and an explicit additional-error constraint. This follow-up is separately logged; no new candidate was added.
