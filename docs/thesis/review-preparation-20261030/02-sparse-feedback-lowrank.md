# Block 2: Sparsification, error feedback and low rank

Reviewed **29 October 2026**; planned Friday block brought forward.
Status: **complete for the scoped evidence notes**. This extends
[Block 1](01-quantization.md). Original papers and the current packet/adapter
code were checked; no new FL run or compressor implementation was started.

## 1. Coordinate-set meaning and method comparison

**Supervisor clarification:** L-Set refers to representing a vector by selecting
a set L of coordinates, such as its largest or most informative entries. This
is a sparsification family description, not an identified standalone algorithm
with a unique paper. Specify the selection rule and cardinality explicitly.
In our project, `k` is an integer count and Top-k means largest magnitude.

| Method | Transmitted representation and state | Assumptions / limitation | Project status |
|---|---|---|---|
| Top-k, [M] | k values and their positions; residual state only if separately enabled | Deterministic magnitude selection is generally biased; contraction does not alone prove local-epoch FL convergence | Global flattened Top-k with persistent client EF implemented |
| Rand-k, [M] | Uniform subset, retained values and positions or synchronized selection information | Unscaled selection is biased; d/k rescaling makes it unbiased but increases variance | Conceptual comparison; not implemented |
| Memory / EF, [M,E] | Adds a dense local residual to a future update; wrapper rather than a standalone wire format | Requires consistent residual units/state and the stated compression/optimization assumptions | Used by Top-k; absent from the QSGD/LLZ/adaptive pilot |
| DGC, [D] | Magnitude-threshold sparsification, accumulated updates and momentum state; packed values/run lengths | Momentum correction, clipping, masking and warm-up are part of the method | Not implemented; current Top-k+EF is not DGC |
| PowerSGD, [P] | Low-rank matrix factors; residual memory and warm-start state | Biased approximation; benefits depend on rank, matrix shape and efficient aggregation | Not implemented; a factor-upload variant would need a separate specification |

Source identifiers resolve in Section 7. A sparse wire message does not imply
sparse working memory. EF, warm starts and optimizer state must be disclosed
when comparing methods.

## 2. Selection and error: derived operator identities

For x in R^d, define a coordinate set L of size k and retain x_j for j in L,
zero elsewhere. The following elementary identities are derived for this
fixed-vector definition; they are not new learning experiments.

Top-k chooses L with largest |x_j|. Its retained energy is at least a k/d
fraction of the total because the mean of the k largest squared entries is
at least the overall mean. Consequently:

```text
||x-Top_k(x)||_2^2 <= (1-k/d)*||x||_2^2
```

For a uniformly selected k-subset without replacement, each coordinate is
retained with probability k/d. Therefore the unscaled Rand-k operator obeys:

```text
E[Rand_k(x)] = (k/d)*x
E[||x-Rand_k(x)||_2^2] = (1-k/d)*||x||_2^2
```

If U(x)=(d/k)*Rand_k(x), direct calculation gives:

```text
E[U(x)] = x
E[||U(x)-x||_2^2] = (d/k-1)*||x||_2^2
```

Unbiasedness and contraction are different properties; rescaling changes the
operator. These distinctions appear in [M, Definitions 2.1-2.2, Section 2.2].
The equations describe numerical operators before float serialization or
protocol overhead. They are not a wall-clock or validation-accuracy ranking.

## 3. What memory/EF changes

[M, Algorithm 1, Theorem 2.4] analyzes memory-equipped compression for smooth
finite-sum components, a strongly convex objective, bounded stochastic-gradient
second moments, a contraction condition and specified steps/iterate averaging.
The sequential theorem does not by itself certify arbitrary asynchronous or
local-epoch FL. Its memory stores step-scaled information when inserted.

[E, Algorithm 2] corrects the next step using the previous residual. Its
smooth nonconvex Theorem II requires compressor contraction (Assumption A),
smoothness (B), and unbiased stochastic gradients with bounded second moments
(C). Its conclusions are conditional, rather than an unconditional guarantee
for every byte codec. The paper's scaled sign compressor also differs from
plain signSGD. Increasing compression can delay the asymptotic regime.

For our **delta-space** convention, define u_i,t as the already step-scaled local
model difference. The implementation's bookkeeping is:

```text
v_i,t = u_i,t + e_i,t                 # corrected delta
c_i,t = decode(encode(v_i,t))         # actual reconstructed delta
e_i,t+1 = v_i,t - c_i,t               # persistent client residual
w_(t+1) = w_t + sum_i a_i*c_i,t
```

This adapts the gradient-step recurrence to model deltas; it is not a new
convergence proof. In exact bookkeeping,
`sum_t c_i,t = sum_t u_i,t + e_i,0 - e_i,T`.
That identity is along the *actual compressed trajectory*: it does not claim
that the u_i,t equal updates from an uncompressed training run. Floating-point
rounding and changes in client weights require further care.

Our [Top-k adapter](../../../federated_compression/topk.py) computes this
residual from the locally decoded packet. It persists in Flower Context across
rounds, validates run/client/layout/round lineage, and rejects missing or stale
state. A model checkpoint alone cannot resume residuals. With float32 dense
state, the residual alone occupies 4d bytes per client, excluding copies and
other model/runtime storage.

For future QSGD+LLZ+EF, measure the residual against the complete decoded
composed compressor. LLZ-only error relative to QSGD would leave first-stage
error uncompensated. Do not mix raw-gradient units with already step-scaled
delta units or apply the learning rate twice. Our additional-LLZ budget uses
a different reference vector from a full-compressor contraction condition;
the old tau=5.9/7.0 values do not establish [E]'s Assumption A.

## 4. Practical systems: DGC and PowerSGD

### DGC

[D, Sections 3.1-3.3, PDF pp. 3-5] accumulates small unsent contributions and
uses magnitude thresholds. Momentum correction changes what is accumulated;
local gradient clipping, momentum-factor masking and warm-up address stability
and stale updates. These mechanisms must accompany a DGC claim. Its sparse
encoding packs float32 values and 16-bit zero-run lengths; roughly 48 bits
per value/run pair excludes overflow handling and framing. Treat this as its
encoding description, not a universal sparse lower bound. Threshold estimation,
packing and dense accumulator/momentum memory cost computation. The reviewed
sections provide empirical design evidence, not a general theorem for our FL.

### PowerSGD

[P, Section 3, Algorithms 1-2, PDF pp. 3-4] reshapes layer gradients into
matrices, uses a power/subspace iteration with orthogonalization, and
communicates factors through two dependent mean all-reduces. Vector parameters
are uncompressed. EF addresses its biased approximation and previous factors
warm-start later iterations. It avoids computing a full SVD; one iteration is
not guaranteed to be the exact best rank-r approximation for every matrix.
Matrix products, orthogonalization, residuals and synchronization matter.
Its data-parallel aggregation differs from our Flower star topology: averaging
independently formed products does not generally equal multiplying averaged
factors. A future FL adaptation must specify shared projections and aggregation,
rather than claim the original method from an arbitrary factor packet.

## 5. Communication accounting: explicit examples

The following cost formulas are derived accounting examples with declared
formats, not universal rates or physical network measurements. Let H include
framing and shape/codec metadata where required.

| Representation | Assumed scalar/index format | Logical content size |
|---|---|---|
| Dense update | d float32 values | 32d bits before H |
| Generic coordinate subset | k float32 values and k fixed-width indices | k*(32+ceil(log2(d))) bits before H/padding |
| Actual project Top-k | float32 values, uint32 indices, 12-byte header | **12+8k bytes** per global packet; Flower metadata additional |
| Generic rank-r matrix factors for n by m matrix | float32 P and Q | **32r(n+m) bits** before H and uncompressed tensors |

In the generic coordinate formula, index bits are packed with the minimum
fixed width; our implementation deliberately uses a whole uint32. Delta/entropy
index coding, a shared mask, or a common seed defines another protocol and must
include its setup/synchronization costs. Actual transport for all-reduce factors
also depends on worker count, collective algorithm and network topology.

For d=4,096 and k=64, our TPK1 packet is **524 bytes**, versus 16,384 raw
FP32 bytes. This was checked against [topk.py](../../../compression/topk.py).
The encoder selects globally across sorted-name flattened tensors, sorts retained
indices, and resolves equal magnitudes deterministically by index. Additional
layout hashes, round identifiers and metrics are in Flower records outside TPK1.

For a purely illustrative n=m=512 matrix and r=1, factors contain 1,024
float32 values versus 262,144 dense values: 4,096 versus 1,048,576 bytes,
a 1/256 content ratio before other costs. This is arithmetic, not an observed
PowerSGD training result. Counting two factor shapes does not mean one
independent server-upload phase reproduces its collective algorithm.

Codec byte savings, total logical Flower communication, CPU cost and physical
network speed are separate measurements. Include warm-up rounds, setup
artifacts, fallback/uncompressed tensors and all training rounds in the chosen
accounting boundary. Keeping residuals locally costs memory but is not an upload
unless the protocol actually transmits them.

## 6. Implications for our comparisons

1. Keep the Top-k+EF label. Existing QSGD/LLZ/adaptive results have no EF;
   their differences do not isolate compressor choice alone.
2. For a later feedback ablation, hold data/model/local work fixed, specify
   residual lifetime and initialization, and use the same complete-compressor
   reference. A benefit is a hypothesis to test, not a promised accuracy recovery.
3. Keep DGC, Rand-k and PowerSGD as literature coverage here. Implementing
   them or reproducing the IEEE paper's low-rank hybrids is separate work.
4. Match actual bytes or report an accuracy/communication/time curve. Equal k,
   rank, or nominal bit depth does not imply equal messages.
5. The literature supports comparing biased compression with memory and
   computational overhead. It does not prove our adaptive budget is safe.

The main table and distinctions can feed the review PDF. Learned/adaptive methods,
recent-work screening and the synthesis are now complete in [Blocks 3-5](progress.md).

## 7. Source record and checked locations

Primary texts retrieved and inspected on 29 October; targeted section review,
not a complete independent proof audit. No access gap blocks this scoped note.
Existing keys resolve in [references.bib](../references.bib).

| ID | Primary source / checked version | Checked locations | Key |
|---|---|---|---|
| M | Stich, Cordonnier, Jaggi, *Sparsified SGD with Memory*, NeurIPS 2018; [author preprint v2](https://arxiv.org/pdf/1809.07599v2) | PDF pp. 3-5: Definitions 2.1-2.2, Sections 2.2-2.3, Algorithm 1, Theorem 2.4 | `stich2018memory` |
| E | Karimireddy et al., ICML 2019, [official PDF](https://proceedings.mlr.press/v97/karimireddy19a/karimireddy19a.pdf) | PDF pp. 4-5: Algorithm 2, Assumptions A-C, Theorem II, Remarks 4-5, Lemma 8 | `karimireddy2019error` |
| D | Lin et al., ICLR 2018; [author preprint v3, revised 2020](https://arxiv.org/pdf/1712.01887v3) | PDF pp. 3-5: Sections 3.1-3.3, Algorithm 1 and encoding description | `lin2018dgc` |
| P | Vogels, Karimireddy, Jaggi, NeurIPS 2019, [official PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/d9fbed9da256e344c1fa46bb46c34c5f-Paper.pdf) | PDF pp. 2-4: Sections 2-4.1, Algorithms 1-2 | `vogels2019powersgd` |

PDFs/hashes and the two small packet-size checks are under ignored
`outputs/literature-blocks-20261029/`. The EF recurrence/assumptions page was
rendered because extraction can lose mathematical glyphs. Full source PDFs and
extracted passages are not included in these versioned review notes. No new
learning result or evidence of measured network speedup was produced.
