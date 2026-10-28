# Adaptive LLZ distortion: focused literature review

Prepared 20 October 2026 after the supervisor meeting, updated 22 October. This
is a focused synthesis, not an exhaustive systematic review. The proposal below
records the original reasoning. The implementation and matched pilot are now
complete, as recorded in the update below.

Implementation update later on 20 October: the 21–22 October work was brought
forward. The [selector contract](adaptive-llz-contract.md) now fixes additional
LLZ error divided by original-update squared norm, with whole-update p=0/1/2
selection. The discussion below records the literature reasoning behind that
choice. The [completed matched pilot](../../outputs/adaptive-study-20261021T104244874038Z/report.md)
now supplies adaptive accuracy and communication results. A convergence proof
for the implemented local-epoch algorithm remains open.

## Research question

How can each client choose an LLZ tolerance each round to reduce the transmitted message size while meeting a specified reconstruction-error constraint, and what conditions would connect that constraint to a learning guarantee?

The supervisor's starting rule is to compute one local update, quantize it once with QSGD, try LLZ p=0/1/2 on the same quantized output, decode locally, and send the smallest acceptable candidate. Varying p is the immediate priority. The prior proposed feedback ablation remains relevant context, rather than replacing this newly agreed direction.

## Evidence map

| Work | Mechanism and evidence | Relevance to our next step | What it does not establish for us |
|---|---|---|---|
| [Alistarh et al., QSGD, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf) | Stochastic quantization with an expected distortion bound and encoding analysis | First-stage quantization error must remain distinct from additional LLZ error | Our dense packet format and local-epoch FedAvg do not inherit every original coding or convergence claim |
| [Zhong et al., LLZ, Globecom Workshops 2023](https://doi.org/10.1109/GCWkshps58843.2023.10464660), and supplied Cao et al. revision | Dictionary coding after quantization, with p controlling signed-symbol tolerance | Defines the LLZ stage and its bounded extra distortion | Fixed-p results do not certify an adaptive client selector or a task-accuracy constraint |
| [Jhunjhunwala et al., AdaQuantFL, ICASSP 2021](https://arxiv.org/abs/2102.04487) | Changes quantization levels during training, relating error to communicated bits | A precedent for time-varying compression and budget-based evaluation | Changes the first-stage quantizer, not LLZ p after a fixed QSGD draw |
| [Hönig et al., DAdaQuant, ICML 2022](https://proceedings.mlr.press/v162/honig22a.html) | Adapts quantization over time and across clients | Especially close motivation for client/round adaptation under an error constraint | Its quantization-variance allocation is not our measured LLZ reconstruction constraint |
| [Reisizadeh et al., FedPAQ, AISTATS 2020](https://proceedings.mlr.press/v108/reisizadeh20a.html) | Local steps, periodic averaging and quantized updates, with convergence analysis | More relevant to local-model updates than a plain distributed-SGD argument | Its unbiased-quantizer assumption cannot simply be assigned to positive-p LLZ |
| [Stich et al., Sparsified SGD with Memory, NeurIPS 2018](https://arxiv.org/abs/1809.07599) | Sparsification with accumulated compression residuals | Context for our Top-k+EF baseline | Does not prove our FedAvg convergence or equate feedback and no-feedback results |
| [Karimireddy et al., Error Feedback, ICML 2019](https://proceedings.mlr.press/v97/karimireddy19a.html) | Analyses of error feedback for biased compression | Bias and feedback must be explicit in the theoretical model | Feedback is not a measured guarantee that our p=1 accuracy will recover |
| [Chandar et al., autoencoder update compression, 2021 preprint](https://arxiv.org/abs/2108.05670) | Learned compression of FL weight updates | Shows learned compression has FL precedents | Our offline candidate is an adaptation and has no FL accuracy result |
| [Isik et al., adaptive compression via side information, AISTATS 2024](https://proceedings.mlr.press/v238/isik24a.html) | Uses server-side distribution information for stochastic compression | Broader context for adaptive coding and later side-information work | It is a different coding construction, not LLZ-SI or our p selector |

## Closest adaptive precedents

**AdaQuantFL** explicitly examines progress against communicated bits. Its Section 3 analyses the quantization/error tradeoff; Section 4 develops a schedule, and Appendix C gives an adaptive-quantization/variable-learning-rate bound. Assumptions include unbiased quantization with a relative variance bound, smooth objectives, bounded stochastic-gradient variance and IID data for that analysis. These are useful proof questions, not conditions we can silently assume for biased LLZ. [Author PDF, Sections 3–4 and Appendix C](https://arxiv.org/pdf/2102.04487)

**DAdaQuant** adapts across clients as well as time. Section 3.5 and Theorem 1 allocate quantization levels using client weights under an expected aggregate quantization-variance constraint. The stated optimization uses a particular stochastic fixed-point/uniform-parameter model, with practical rounding. It should not be presented as an unrestricted convergence guarantee. Our clients currently have equal training counts; measured update content can still give different LLZ choices. [Author PDF, Sections 3.4–3.6](https://proceedings.mlr.press/v162/honig22a/honig22a.pdf)

These precedents rule out claiming that adaptive FL compression or client-specific precision is itself new. A candidate thesis contribution is the narrower LLZ-specific selector, its distortion-budget design and an analysis valid for its actual assumptions. This search does not establish priority for that exact combination.

## LLZ source and proof boundary

Our algorithm reference is the supplied **Cao et al. revision-1 manuscript**, *Federated Learning with Lossy Lempel-Ziv Gradient Compression*, generated 2 June 2026. It is not cited as an accepted journal article. Reuse the inspected clean manuscript and [paper-to-code audit](llz-paper-audit.md), including its reconstructed-dictionary interpretation. [Source provenance and notation](../../papers/README.md) records the PDF hash and the conversion between the manuscript's resolution and project levels=127.

In that source, p bounds errors in signed integer symbols. For one tensor with QSGD norm r, resolution levels, and m mismatched symbols, the audit maps the additional squared reconstruction-error bound to m*(p*r/levels)^2 before final dtype rounding. The same p can therefore have different effects in different tensors or rounds. The manuscript's gradient-update convergence analysis is not already a proof for our local-epoch model-delta pipeline or data-dependent p selection.

**FedPAQ** is a useful local-update theory starting point. Its Assumption 1 requires an unbiased quantizer with norm-relative variance, and the main setup considers common IID client distributions. Smoothness and unbiased, variance-bounded stochastic gradients are also explicit. Positive-p LLZ can introduce bias, so importing its theorem unchanged would leave a missing hypothesis. [Author PDF, problem setup and Section 4](https://proceedings.mlr.press/v108/reisizadeh20a/reisizadeh20a.pdf)

## Proposed distortion formulation to discuss

Let u be one client update, z its decoded QSGD reconstruction from a **single fixed quantization draw**, and z_p the locally decoded LLZ candidate. Distinguish:

- Additional LLZ error: d_LLZ(p) = ||z_p - z||_2^2.
- Total compression error: d_total(p) = ||z_p - u||_2^2.

The two quantities answer different questions. A working *proposal*, not an agreed threshold, is to constrain the additional error by d_LLZ(p) <= tau_(client,round)*||u||_2^2. An absolute budget or normalization by ||z|| instead would define a different protocol. Zero updates need an explicit rule without hiding a division-by-zero adjustment inside the theory.

For each p in {0,1,2}, measure decoded error and actual transmitted bytes, including tolerance/framing metadata. Choose the minimum-size feasible candidate, with a deterministic tie rule. Do not assume that increasing p always produces the smallest actual packet. First decide whether p is selected for the whole client update or per tensor; the notes suggest starting at client/update level. A per-tensor extension would need its own budget-allocation rule.

Under an additional-LLZ-error constraint, p=0 is a feasible fallback because it adds no distortion to z. Under a total-error constraint, even p=0 may violate the limit due to QSGD's error. A safe fallback cannot be described without specifying which constraint we use.

The basic decomposition ||z_p-u|| <= ||z-u|| + ||z_p-z|| explains how the two stages interact. It is an elementary norm inequality, **not a convergence theorem**. Even if QSGD is conditionally unbiased in its ideal model, LLZ selection need not preserve that property. Any learning analysis must explicitly handle the composed compressor's bias and local-client drift.

## What our existing results contribute

The [three-seed study](../../outputs/compression-repeats-20261016T105615971335Z/analysis/report.md) found exact p=0 replay in all seeds, with mean selected accuracy 83.73%, matching QSGD. Fixed p=1 averaged 78.67%, a paired loss of 5.07 pp, despite smaller uploads. This motivates investigating controlled distortion. It does not demonstrate that adaptive p succeeds. Top-k+EF is a completed FL baseline; feedback differs from QSGD/LLZ.

The [AE feasibility decision](../../outputs/ae-feasibility-20261006T091559343464Z/decision.md) records about 6.42% projected codec-only saving against standalone Top-k after decoder setup, with about 2.51 times its reconstruction error and higher CPU cost. The AE is implemented offline but not integrated into FL. This is an implementation-status distinction to state directly to the supervisor.

## Theory work still required

| Question | Needed before claiming a learning guarantee |
|---|---|
| What converges? | Choose an objective-gap or stationarity statement. Recognition accuracy is a separate empirical outcome. |
| What is compressed? | Match the proof to stochastic gradients or local model deltas, including local steps and client weights. |
| How does selection depend on data? | Express the conditional error/bias guarantee given the current state and update. |
| What happens to distortion over time? | Determine sufficient budget/learning-rate conditions; a fixed positive tolerance cannot simply be declared safe. |
| Does feedback exist? | Analyse the stated residual recursion if used. Keep it distinct from the no-feedback setup. |
| Which assumptions hold? | State smoothness, noise, heterogeneity and participation assumptions explicitly; do not claim neural-network convexity. |

The initial theoretical milestone is complete: an assumption map, a deterministic
aggregation bound, a conditional composition/bias bound and a descent step.
The [22 October extension](adaptive-llz-conditional-stationarity.md) derives
conditional stationarity bounds and a one-full-gradient-step special case.
The assumptions connecting actual local-epoch deltas to global descent remain
unproved. This review does not launch a new experiment.

## Completed adaptive pilot and its literature implications

The matched seed-42, 15-round pilot used frozen budgets 5.9 and 7.0, calibrated
from eight saved training-image-only deltas before new validation outcomes.
The tighter policy achieved 83.2% selected validation accuracy versus 83.4%
for lossless LLZ, with 35.45% fewer training-upload bytes. This is a descriptive
tradeoff, not evidence of equivalent accuracy. The looser policy achieved
78.5% with 13.278 MB uploads, compared with fixed p=1 at 78.7% and 12.483 MB.
It did not improve that control's accuracy/upload point in this pilot.

These observations support reporting a rate-distortion-learning tradeoff,
as motivated by adaptive quantization, rather than assuming maximum permitted
distortion is best. Minimizing current-message bytes does not minimize total
bytes along a different future learning trajectory. The tighter policy reduced
total logical serialized bytes versus lossless LLZ by only 0.514%, and its
successful launch took 79.1 minutes versus 32.3. Downloads and validation
exchanges remain uncompressed. Measurements include metadata and diagnostics
but are not captured network traffic. All 60 client-round choices per adaptive
policy passed the saved feasibility/minimum-byte audit. No test-set evaluation
or additional seed is included in this adaptive study.

The defensible research gap remains a second-stage LLZ selector under measured
whole-update distortion and complete-message cost, including its computation
cost and learning assumptions. This does not establish priority, optimal
budgets or a completed convergence theorem. The larger pilot budgets also
fall outside the simple tau<1 sufficient condition in the one-step result.

## How to present the research gap

“Adaptive quantization already exists in federated learning. Our next question is how to adapt the distortion of a second LLZ stage after a fixed QSGD draw. We will choose among actual packets under a measured error budget, then investigate which budget conditions support a learning guarantee for the chosen update rule.”

For the meeting use the [short literature briefing](adaptive-llz-meeting-brief.md). Bibliographic keys for the added works are in [references.bib](references.bib).

## Reading coverage

The QSGD/LLZ/feedback background reuses the earlier review and supplied-paper audit. This update verified official records for DAdaQuant, AdaQuantFL and FedPAQ and inspected the source sections identified above. The autoencoder and side-information entries are contextual records/abstract-level checks here, not fresh reproductions or full new theorem audits. No exhaustive “first method” or acceptance-status claim follows from this bounded search.
