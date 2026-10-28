# Communication-efficient federated learning with QSGD and LLZ-p

For the 20 October supervisor direction, see the
[adaptive-LLZ literature update](adaptive-llz-literature-review.md). This
September review remains historical background; its experiment and future-work
status should not replace the newer three-seed evidence and AE feasibility result.

Focused literature review, 25 September 2026. Companion:
[experimental design](experimental-design.md). Citation keys correspond to
[references.bib](references.bib).

## 1. Scope and motivation

The project investigates whether a second coding stage can reduce communication
beyond quantization while retaining useful learning performance. It uses facial
identity classification as an experimental task, rather than claiming a new
face-recognition architecture. The review covers local training, quantization,
dictionary coding, sparsification, and error feedback. Sources were selected
for their direct relationship to QSGD and the two supplied LLZ papers; this is
not an exhaustive survey or a claim that these are all current alternatives.

Federated learning trains a shared model using updates computed from distributed
data. Its practical constraints include communication, statistical heterogeneity,
and device/system differences. Keeping raw data local is not itself a formal
privacy guarantee. A simulation isolates some of these factors but does not
establish deployment performance. [Kairouz et al., 2021 (`kairouz2021advances`)](https://arxiv.org/abs/1912.04977)

Two interventions address different parts of communication cost: perform more
local computation between exchanges, or send fewer bits at each exchange.
FedAvg combines local training with server-side model averaging. Its local-step
approach motivates holding the training schedule fixed when evaluating a codec;
otherwise a change in local computation can be confused with a compression
benefit. [McMahan et al., 2017 (`mcmahan2017fedavg`)](https://proceedings.mlr.press/v54/mcmahan17a.html)

For this project, communication has three distinct measurement levels:
standalone tensor packets; logical serialized Flower messages; and physical
network traffic. Only the first two are currently measured. Total message bytes
include both training and validation exchanges, so upload savings alone cannot
describe end-to-end savings. This is an accounting choice of our implementation,
not a claim about a universal FL protocol. See the protocol for exact boundaries.

## 2. QSGD: quantization and encoding are different operations

QSGD stochastically rounds normalized coordinate magnitudes to discrete levels
and retains their signs. For a vector x, norm r, and s intervals, set
`a_i = s |x_i| / r`; round `a_i` to its adjacent integers with probabilities
preserving its expectation, then reconstruct `r sign(x_i) level_i / s`.
Zero vectors reconstruct as zero. In exact arithmetic the quantizer is unbiased,
with expected squared error bounded by
`min(d/s^2, sqrt(d)/s) ||x||_2^2`. This is an expectation bound, not a bound on
every random realization. The original work also specifies efficient encoding
and analyzes distributed SGD under stated assumptions.
[Alistarh et al., 2017, Section 3.1 (`alistarh2017qsgd`)](https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf)

Our implementation uses that stochastic quantizer with one norm per tensor,
but its `QSD1` packet uses dense fixed-width sign/magnitude coding rather than
the paper's Elias-based encoding. At `levels=127`, it spends eight code bits per
coordinate plus metadata. Consequently, our measured QSGD packet sizes must not
be described as reproducing the paper's coding-rate guarantees. Nor does the
paper's SGD analysis automatically prove convergence for our multi-minibatch
FedAvg model-delta pipeline. These implementation distinctions are documented
in [the codec specification](../../compression/README.md).

## 3. LLZ-p as a second compression stage

Zhong, Liang, and Wu introduce lossy Lempel-Ziv coding as an additional stage
after an initial compression such as QSGD. Rather than sending every quantized
symbol independently, the method represents repeated or approximately repeated
subsequences using dictionary references and literals. The tolerance p trades
reconstruction fidelity for opportunities to match longer subsequences. The
conference paper reports favorable comparisons under its experimental setting;
those observations do not establish that one tolerance works for every task.
[Zhong et al., 2023 (`zhong2023llz`)](https://doi.org/10.1109/GCWkshps58843.2023.10464660)

The supplied revision by Cao et al. clarifies the symbol pipeline, dictionary
synchronization, and the following literal, and adds LLZ-SI and broader
experiments. It is a revision-1 submission bundle generated on 2 June 2026,
not a publication whose acceptance has been verified. The clean manuscript is
PDF pages 4-18. This review uses that portion, not the later tracked revisions
or response material. [`cao2026llzmanuscript`; source provenance](../../papers/README.md)

### Meaning of p and synchronization

In the revision, p bounds the absolute difference between original and decoded
**signed integer quantization symbols**. It is neither an accuracy percentage
nor a fixed tolerance in floating-point model units. The encoder must build
its dictionary from reconstructed history so the decoder can reproduce the same
references. A phrase contains a position, match length, and explicit next
literal. See Section III, Equation (5), and Algorithms 2-3 (PDF pages 7-9).

The notation differs across sources: the revision uses `s-1` quantization
intervals, while our `levels` counts the intervals themselves. Thus project
`levels=127` corresponds to revision `s=128`, with 255 signed symbols from
-127 through 127. This conversion is essential when comparing configurations.

Let q be the QSGD symbols, q-hat their LLZ reconstruction, and m the number of
mismatched coordinates in one tensor. With norm r retained from QSGD and
`|q-hat_i - q_i| <= p`, the additional dequantized squared error is at most
`m (p r / levels)^2` before output floating-point rounding. This is the
per-tensor interpretation of the revision's Lemma 1, Equation (8), PDF page 12.
The same p therefore has a different floating-point effect when the tensor norm
or quantization resolution changes.

At p=0, the LLZ stage must reconstruct the QSGD symbols exactly. It removes
redundancy without adding quantization distortion; it does **not** reverse
QSGD's earlier loss. At p>0, matching can introduce bias. Lower packet size is
not evidence that training quality is preserved.

Short or poorly matching sequences can expand because triplets and headers
cost bits. Our packets have no automatic raw fallback. Compression benefit is
therefore a measured property of the input and settings, not a guaranteed
consequence of enabling LLZ.

### Limits of transferring the paper's conclusions

The revised theorem analyzes QSGD-LLZ-p without error feedback, under smoothness,
bounded-gradient/variance, unbiased-gradient, and quantizer assumptions with a
specified step-size condition. It addresses stochastic-gradient updates at the
current iterate. Our clients instead take a local epoch of SGD and transmit
model deltas. We should not cite the theorem as a convergence proof for our
implementation. The revision treats LLZ-SI as an empirical extension rather
than including it in that theorem (Section V, PDF pages 11-13).

The empirical setup differs again: Section VI (PDF page 14) says that error
feedback is used unless otherwise stated, reports CIFAR tasks, learning-rate
decay, and longer training, and defines a transmitted-bit ratio relative to raw
FP32 gradients. Our DigiFace pilot uses no error feedback, 15 fine-tuning rounds,
fixed learning rates, model deltas, and a logical-message total that includes
downloads and validation. Its p=1 accuracy loss is therefore not a contradiction
of the paper's experiments. We have not isolated which difference explains it.

The revision reports both IID and non-IID experiments. The thesis must not
claim that non-IID evaluation is absent from the supplied literature. Its
opportunity is a controlled, reproducible study in our different setting.

## 4. Relevant alternatives and what a fair comparison requires

| Approach | Main mechanism | Relevance and comparison requirement |
|---|---|---|
| Uncompressed FedAvg | Local training, then model averaging | Fixed training control for the present study |
| QSGD | Stochastic quantization | Isolates the first compression stage; identify its actual byte encoding |
| QSGD + LLZ-p | Dictionary coding over quantized symbols | Isolates incremental redundancy removal and the cost of tolerance |
| Top-k sparsification | Retain the largest-magnitude coordinates | Count both values and coordinate indices; specify residual/error-feedback policy |
| Error feedback | Carry reconstruction residual into later updates | An algorithmic factor to control, not a free implementation detail |
| signSGD | Sign-based communication | Useful conceptual extreme; sign aggregation changes the optimizer |
| DGC | Sparsification with additional training mechanisms | A fuller system baseline, not interchangeable with plain Top-k |
| PowerSGD | Low-rank approximation of gradient matrices | Relevant to computation and wall-clock trade-offs; not implemented here |
| LLZ-SI | Dictionary coding using shared historical information | A later extension with explicit synchronization and state requirements |

Sparsified SGD with Memory analyzes sparse updates with error compensation,
including Top-k-style selection. That support does not automatically apply to
a stateless Top-k implementation. Any later comparison must count sparse
indices and distinguish Top-k with and without memory.
[Stich et al., 2018 (`stich2018memory`)](https://proceedings.neurips.cc/paper/2018/hash/b440509a0106086a67bc2ea9df0a1dab-Abstract.html)

Error feedback stores the difference between an update and its compressed
reconstruction, adding it to a later update. Karimireddy et al. demonstrate
failures of sign compression and analyze how feedback can address them under
their framework. This motivates a future controlled feedback ablation, not a
promise that feedback will eliminate our p=1 accuracy loss.
[Karimireddy et al., 2019 (`karimireddy2019error`)](https://proceedings.mlr.press/v97/karimireddy19a.html)

signSGD communicates signs and uses a different aggregation/update construction;
it is informative background but is not a drop-in codec comparison with our
FedAvg implementation. [Bernstein et al., 2018 (`bernstein2018signsgd`)](https://proceedings.mlr.press/v80/bernstein18a.html)

DGC combines compression with momentum correction, local gradient clipping,
momentum-factor masking, and warm-up. Its published compression results cannot
be attributed to naive magnitude pruning alone. [Lin et al., ICLR 2018
(`lin2018dgc`)](https://arxiv.org/abs/1712.01887)

PowerSGD uses efficient low-rank approximations and emphasizes practical
aggregation and measured training speed. It reinforces the need to assess
computation as well as bytes, though its distributed setting is not identical
to this Flower simulation. [Vogels et al., 2019 (`vogels2019powersgd`)](https://proceedings.neurips.cc/paper_files/paper/2019/hash/d9fbed9da256e344c1fa46bb46c34c5f-Abstract.html)

The revised LLZ manuscript's Section IV uses historical aggregated updates as
side information for LLZ-SI. In our project that extension would need explicit
state availability, quantization, initialization, and restart rules. It is
outside the present implementation milestone. [`cao2026llzmanuscript`](../../papers/README.md)

## 5. Experimental task and current evidence

DigiFace-1M supplies synthetic face images rendered for face-recognition
research. The original paper studies recognition and synthetic-to-real
generalization; those benchmark results are not directly comparable with
our 100-way, closed-set identity classification. Our images in validation/test
are distinct, but their identities are also present in training. This does
not evaluate verification or recognition of unseen identities.
[Bae et al., WACV 2023 (`bae2023digiface`)](https://openaccess.thecvf.com/content/WACV2023/html/Bae_DigiFace-1M_1_Million_Digital_Face_Images_for_Face_Recognition_WACV_2023_paper.html)

The completed seed-42 fine-tuning pilot selected 83.3% validation accuracy for
uncompressed training, 83.4% for QSGD and p=0, and 78.7% for p=1. LLZ p=0
replayed all 15 QSGD rounds exactly while reducing QSGD training uploads by
88.21%. Total logical-message bytes fell by 9.79%, because other traffic stayed
large. Relative to p=0, p=1 saved another 1.15% of total bytes and lost 4.7
accuracy percentage points. These are project results, not literature claims;
the source table and hashes are in the [protocol](experimental-design.md).

The pilot supports implementation correctness and a promising operating point.
It does not establish statistically equivalent accuracy, convergence, real-network
speedup, or a universal failure of positive p. The 0.1-point accuracy difference
between QSGD and the control is one image in the 1,000-image validation set.

## 6. Research gap and thesis contribution

**Working research question:** Under a controlled FedAvg model-delta protocol,
how much incremental communication reduction does LLZ-p provide beyond QSGD,
and what accuracy and computation costs arise as distortion and data
heterogeneity change?

This is a gap in our current evidence and an application-specific empirical
question, not a claim that no prior work has studied these topics. The literature
already contains LLZ-p, feedback mechanisms, and non-IID comparisons. Simply
integrating them does not establish a new compression algorithm.

A defensible master's contribution is a paper-traceable codec implementation
and a controlled characterization of its behavior on model deltas: exact p=0
replay, positive-p distortion measured separately from QSGD, explicit metadata
accounting, and separation of upload savings from total messages and computation.
Later work can ask whether feedback or heterogeneous partitions change the
trade-off; those are proposed investigations, not completed findings.

Experimental controls and transparent simulation assumptions are central to
making such results interpretable. [Wang et al., 2021
(`wang2021fieldguide`)](https://arxiv.org/abs/2107.06917)

The completed [paper-to-code audit](llz-paper-audit.md) records the interpretation
used by this implementation. In particular, Algorithm 2 initializes reconstructed history but uses
`g` in some dictionary expressions; the accompanying prose and worked example
explicitly use reconstructed symbols. Our existing implementation follows that
interpretation. The deterministic tie rule follows ascending enumeration and
last-match overwrite in the revised algorithm. Complete packet framing is a
project choice, recorded in the audit and [existing mapping](../../papers/README.md),
and should not be attributed to an author-provided wire format.

## 7. Source record and reading priorities

Public links above identify the original papers or authors' preprints. The
conference LLZ DOI was not retrievable through the browser in this session;
its content was checked from the supplied six-page PDF. Both local PDFs match
the SHA256 fingerprints already recorded in `papers/README.md`. The revised
manuscript's Algorithms 2-3 and Lemma 1 were also inspected as rendered pages.

For reviewing and maintaining the implementation, prioritize the revision's Section II (notation),
Section III and Algorithms 2-3 (codec), Section V (assumptions and error), and
Section VI (feedback and measurement). Use the conference version for historical
context and report the revision used. Do not silently mix their pseudocode or
quantization notation. Bibliographic entries for the two large overview papers
abbreviate their author lists with `and others`; expand these if required by the
final thesis style.
