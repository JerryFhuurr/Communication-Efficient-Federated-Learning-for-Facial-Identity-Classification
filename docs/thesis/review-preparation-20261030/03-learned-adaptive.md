# Block 3: Learned and adaptive compression

Reviewed **29 October 2026**; planned Friday work brought forward.
Status: **complete for scoped evidence notes**. Four primary PDFs were checked
at the locations below. Claims describe those sources; project implications
are our interpretation. No new experiments or independent proof audit.

## 1. Attributed comparison

| Method / source | What changes and what is transmitted | State, costs and assumptions | Relation to adaptive LLZ |
|---|---|---|---|
| Chandar et al. (2021), [A] | Train a client autoencoder on task-model weight trajectories; transmit encoded representations and give the server a decoder. | Local pre-training, AE fitting and decoder transfer; reconstruction evidence does not establish a general FL convergence theorem. | Learned update compression already exists. Our standalone AE is an adaptation, not a new method or an integrated FL result. |
| AdaQuantFL (2021), [B] | Adjust stochastic quantization levels over training; transmit quantized local model differences. | Unbiased relative-variance quantizer, smoothness, bounded stochastic variance and common IID setting; theory optimizes a learning bound under a bit budget. | Precedent for adapting precision, whereas our first-stage QSGD levels stay fixed. |
| DAdaQuant (2022), [C] | Adapt quantization over time and across differently weighted clients; encode local parameter differences. | Loss-history state and client weights; allocation uses a quantization-error model and a communication proxy. | Equal-sized clients remove its weight-driven allocation differences; LLZ can still exploit different update contents. |
| Isik et al., KLMS (2024), [D] | Use shared side-information distributions and randomness to communicate a sample by its index in a common candidate set. | Shared distribution/seed, block partitions and refresh messages; distributional approximation rather than exact recovery of an arbitrary pre-existing sample. | Historical information can reduce coding cost, but this is distinct from LLZ's approximate dictionary matching. |

The four methods answer different questions. A smaller latent dimension, a
quantizer variance bound, a sampling guarantee and an LLZ symbol error bound
are not interchangeable learning-quality guarantees.

## 2. What the checked sources establish

**Autoencoder precedent [A, PDF pp. 2-4, Sections 3-4].** The proposal first
collects intermediate task-model weights, fits a client-specific AE and sends
the decoder to the server. Its empirical workflow reconstructs saved classifier
weights and assesses reconstruction/classifier quality. Treat this as
proof-of-concept evidence, rather than our matched closed-loop FedAvg benchmark.
The paper explicitly discusses the large fully connected AE and decoder
transfer burden; a dimensional compression ratio is not a complete byte ledger.

**Adaptive quantization [B, PDF pp. 2-3, Sections 3-4; Appendix C].** The
idealized per-client message cost is
`d*ceil(log2(s+1)) + d + 32` bits, including a norm but excluding our Flower
framing. The practical policy increases levels approximately with the inverse
square root of training loss, with an initial level chosen separately. It uses
an approximation to unknown loss/optimization quantities. An extension also
accounts for changing learning rates. This is a conditional bound-guided
precision rule, not an optimal measured-packet selector or a universal safe
reconstruction threshold.

**Doubly adaptive quantization [C, PDF pp. 4-5, Sections 3.3-3.6].** A smoothed
training-loss plateau triggers increased precision, bounded by a maximum.
Across clients, levels scale with aggregation weight to the power two-thirds,
then are rounded/clamped. The allocation minimizes a sum-of-levels cost proxy
while matching a reference aggregate quantization variance. Its derivation
uses an unbiased fixed-point model with scalar parameters uniformly distributed
on a bounded interval. That allocation result is not a blanket theorem for our
biased second-stage coder or complete serialized-message costs.

**Side information [D, PDF pp. 2, 5-6, Sections 4.1-4.2, Theorem 4.1].** A shared
seed generates candidates from distribution P; the client uses Q/P weights and
communicates an index. Candidate count scales exponentially with KL divergence;
index length scales with its logarithm. The theorem controls expectations under
its density-ratio/integrability conditions, not exact replay of an already drawn
QSGD vector. Blocking makes the candidate search practical; adaptive boundaries
and their refresh require communication and synchronized state.

## 3. Decoder, setup and state accounting

For our comparisons, use an explicit accounting model rather than comparing
only latent length or selected values:

```text
B_total = B_initialization + B_calibration_transfer + B_decoder_distribution
        + sum_rounds(B_selected_upload + B_download + B_control
                     + B_decoder_or_shared_state_refresh)
```

Count each actual transfer according to the chosen ledger boundary. Local
calibration data that never leaves a client costs computation/storage, not
network bytes. Decoder files sent once and cached should be counted once;
re-sent or updated versions cost again. Shared seeds are cheap but do not make
distribution parameters, partitions, layout or version identifiers free.

Record calibration/fitting, candidate encoding, local candidate decoding,
selection time and server decoding separately. Only the chosen candidate is
transmitted by our selector; rejected candidates still cost client computation.
A learned decoder additionally needs an architecture/version, tensor layout,
normalization and numeric representation. Floating-point latent values are
not entropy-coded symbols merely because the encoder is a neural network.
Any model updates needed to distribute a new decoder must be charged to the
method that requires them.

These are proposed comparison rules, not claims that the four sources use one
common complete packet format. Codec bytes, serialized Flower objects and
physical traffic remain different denominators.

## 4. Existing project AE: an offline feasibility result

The saved [6 October decision](../../../outputs/ae-feasibility-20261006T091559343464Z/decision.md)
records a standalone block AE with architecture 256-64-6-64-256. It was **not
integrated into Flower**. On those saved update probes, constructed codec
bundles averaged 924,310 bytes/update versus Top-k's 992,594; pooled relative
squared errors were 0.665316 versus 0.264914. Including 273,556 decoder setup
bytes across four clients, the 60-upload codec-only projection saved 6.42%.
That modest size benefit accompanied worse reconstruction and higher measured
CPU encode/decode times in this implementation.

Those numbers support retaining a negative feasibility result for this AE.
They establish neither an identification-accuracy comparison nor the
inferiority of learned compression generally. Reconstruction validation also
selected the checkpoints, and local trajectories shared the training-image
population; there was no untouched test-set FL evaluation for this codec.
The decision's old request to clarify L-Set has since been resolved as
coordinate selection; see [Block 2](02-sparse-feedback-lowrank.md).

## 5. Mapping to our LLZ question

The [IEEE version comparison](../llz-ieee-version-comparison-2026-10-29.md)
separates published LLZ mechanisms from our implementation and possible
extension. LLZ-SI uses decoder-side sequence/dictionary information; KLMS's
shared probability distributions and common-randomness sampling are a different
mechanism. We have not implemented either side-information variant.

Current behavior was checked in
[the independent selector](../../../compression/adaptive_llz.py) and
[the Flower adapter](../../../federated_compression/adaptive_llz.py): one QSGD
draw per tensor is reused for p in {0,1,2}. If u is the original update, z its
decoded QSGD representation and z_p the decoded LLZ result, selection enforces

```text
E_extra(p) = sum_tensors ||z_p - z||_2^2
N_original = sum_tensors ||u||_2^2
E_extra(p) <= tau * N_original
p_selected = argmin_feasible (complete_serialized_reply_bytes, p)
```

The second tuple entry breaks byte ties toward lower p. A zero original norm
or zero budget requires zero additional error. p=0 is a checked lossless
fallback relative to QSGD reconstruction, not relative to the original float
update. In standalone use, the selector defaults to packet-byte costs; the
Flower adapter supplies complete reply costs including diagnostics. The server
checks consistency but cannot independently measure error against the client's
unavailable original update; honest measurement remains an assumption.

**Research interpretation.** Unlike [B-C], we adapt a second-stage code after
fixing first-stage quantization. Unlike [D], we preserve one common QSGD draw
when testing candidates. These differences help define the experimental
question; they do not establish novelty or convergence. [Block 4](04-recent-work-search.md)
checks newer alternatives before any priority claim.

Our budget bounds *additional* LLZ squared error divided by original update
squared norm. It does not bound total QSGD-plus-LLZ distortion, loss increase or
validation degradation. The triangle inequality gives a useful deterministic
accounting statement, with all tensors concatenated:

```text
||z_p-u||_2 <= ||z-u||_2 + sqrt(tau)*||u||_2
```

This does not imply conditional unbiasedness or contraction below one.
Historical values 5.9 and 7 were pilot heuristics, not percentages or
learning-safe constants. Budget selection remains a scientific task: define a
validation-quality tolerance and communication objective before selecting a
budget using development evidence, then assess that choice on held-out runs.
Do not choose it using final test accuracy. No new budget is fixed here.

## 6. Source register and handoff

| ID | Checked primary version | Locations | Existing bibliography key |
|---|---|---|---|
| A | Chandar et al., 2021 [author preprint v1](https://arxiv.org/pdf/2108.05670v1) | PDF pp. 2-4; pipeline, experimental workflow and decoder-size discussion | `chandar2021autoencoder` |
| B | Jhunjhunwala et al., 2021 [author preprint v1](https://arxiv.org/pdf/2102.04487v1), records ICASSP acceptance | PDF pp. 2-3; assumptions, bit model and practical policy; Appendix C extension | `jhunjhunwala2021adaquantfl` |
| C | Honig et al., ICML 2022 [official PDF](https://proceedings.mlr.press/v162/honig22a/honig22a.pdf) | PDF pp. 4-5; Sections 3.3-3.6 | `honig2022dadaquant` |
| D | Isik et al., AISTATS 2024 [official PDF](https://proceedings.mlr.press/v238/isik24a/isik24a.pdf) | PDF pp. 2, 5-6; Sections 4.1-4.2, Theorem 4.1 | `isik2024sideinformation` |

Exact URLs, SHA256 fingerprints and extracted text are retained in ignored
`outputs/literature-blocks-20261029/`. No access gap blocks these four scoped
readings. The [bibliography](../references.bib) already contains their keys.
For Block 5, merge the rows with Blocks 1-2, preserve the offline-AE label and
carry the unresolved budget/theory questions into the review. Do not describe
this note as new experimental validation or a final literature-review PDF.
