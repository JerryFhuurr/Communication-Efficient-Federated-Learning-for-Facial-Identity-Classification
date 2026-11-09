# Block 1: Quantization and FL update foundations

Reviewed **29 October 2026**; planned Friday block brought forward.
Status: **complete for the scoped evidence notes**. Targeted full-text checks
covered definitions, assumptions, algorithms and communication descriptions;
this is not an independent validation of every proof or a new experiment.

## 1. Attributed comparison

Use the sources and exact locations in Section 6. An unbiased *quantizer*
does not make a multi-step *local update* an unbiased gradient at the global
model. These are different statements.

| Method | Representation / learning setting | Assumptions and claim boundary | Relevance to our thesis |
|---|---|---|---|
| QSGD (2017), [Q] | Norm and signed stochastic levels; distributed SGD. Original encoding exploits nonzero gaps and magnitudes using Elias codes. | Conditional unbiasedness and relative variance bound; optimization results require their SGD assumptions. Theoretical encoding lengths belong to that encoding. | First-stage reference; our dense encoding needs its own measured bytes. |
| signSGD (2018), [S] | Coordinate signs; distributed majority voting can return signs to workers. | Lower-bounded objective, coordinate smoothness, unbiased independent gradient estimates and coordinate variance bounds. Improved worker scaling adds symmetric unimodal noise; theorem uses specified batch/step schedules. | Changes direction/aggregation and discards magnitude; it is not a drop-in FedAvg codec. |
| FedPAQ (2020), [F] | Quantized model differences after local steps, periodic averaging and uniformly sampled partial participation. | Common IID client distribution; smooth objectives, unbiased variance-bounded gradients and an unbiased relative-variance quantizer. Strong convexity is added for its strongly convex result. | A closer local-update precedent; positive-p LLZ and our precise local-epoch protocol need separate analysis. |

These rows compare mechanisms and assumptions, not benchmark accuracy across
unrelated datasets. EF is treated separately in [Block 2](02-sparse-feedback-lowrank.md).

## 2. Gradient versus model delta: consistent notation

The following equations fix the notation for this review and our project.
Let the global model be w_t, client i's local iterates be w_(i,t,h), and
aggregation weights a_i sum to one.

**One gradient evaluated at the broadcast model** gives

```text
g_i,t = stochastic_gradient_i(w_t)
w_(t+1) = w_t - eta_t * sum_i a_i * C(g_i,t)
```

**A local trajectory followed by a transmitted model difference** gives

```text
w_(i,t,0) = w_t
w_(i,t,h+1) = w_(i,t,h) - eta_(i,t,h) * g_i(w_(i,t,h))
u_i,t = w_(i,t,H) - w_t
w_(t+1) = w_t + sum_i a_i * C(u_i,t)
```

For plain SGD, expanding the local recursion yields
`u_i,t = -sum_h eta_(i,t,h) * g_i(w_(i,t,h))`.
With one step and aligned settings, this reduces to a scaled negative gradient.
With several steps, gradients are evaluated at different local models. Applying
an unbiased codec to u preserves u in conditional expectation; it does not
remove the local-trajectory effect. FedPAQ discusses this distinction explicitly
in Section 4 (PDF p. 6). Extra optimizer terms would require a revised expansion.

The adapter [updates.py](../../../federated_compression/updates.py) actually
forms `trained - reference`. Use the plus sign for the reconstructed delta at
the server; do not multiply it by a learning rate a second time. Local training
already used its learning rate. Our one local epoch includes multiple minibatch
steps, rather than a single stochastic gradient evaluation.

In this project, equal-size four-client IID partitions give equal aggregation
weights. Weighted aggregation, changing participation, fixed BatchNorm buffers
and per-tensor normalization still belong in the experiment specification.
A simulated IID split does not verify every stochastic-oracle assumption of a
published theorem.

## 3. Quantization, encoding and guarantees

### QSGD: separate numerical reconstruction from byte representation

For x != 0, set r=||x||_2, l_j=floor(s*|x_j|/r), and
q_j=sign(x_j)*(l_j+Bernoulli(s*|x_j|/r-l_j)). Reconstruct
`Q_s(x)_j = r*q_j/s`; zero inputs map to zero. In ideal arithmetic [Q,
Section 3.1, Eq. (4), Lemma 3.1]:

```text
E[Q_s(x) | x] = x
E[||Q_s(x)-x||_2^2 | x] <= min(d/s^2, sqrt(d)/s) * ||x||_2^2
```

The variance statement is an expectation bound, not a per-draw limit. The
paper's source encoder is a second operation, distinct from stochastic
rounding. Changing the lossless byte encoding preserves a quantized
reconstruction but changes communication cost. Finite precision needs its own
qualification.

### signSGD: signed optimization

[S, Sections 3-4, Theorems 1-2] analyzes sign-based optimization and majority
voting under its stated schedules and assumptions. A sign is generally a biased
representation of a real gradient. One sign bit per coordinate assumes a shared
binary sign/tie convention; shape/framing remain additional costs. A separate
ternary zero symbol needs a different encoding. Do not equate signSGD with
QSGD at s=1: QSGD also stochastically outputs zero and retains a norm.

### FedPAQ: local computation and compressed communication

[F, Sections 2-4, Algorithm 1, Eq. (6), Assumptions 1-3] quantizes the local
model difference and adds the averaged reconstructed differences to the server
model. It studies strongly convex and nonconvex settings with prescribed
step-size/local-period conditions. It does not prove our adaptive LLZ method:
positive-p reconstruction can be biased, and its assumptions are not established
for our exact implementation. Its quantizer abstraction does not define a
complete Flower packet format.

## 4. Actual project packet costs

These are implementation-derived calculations, not encoding-rate claims of [Q].
Let d be one tensor's coordinate count, rnk its number of shape dimensions and
s the project's `levels`. From [qsgd.py](../../../compression/qsgd.py), `QSD1` has:

```text
header = 16 bytes                 # version, dtype, ndim, levels, float64 norm
shape = 4*rnk bytes               # uint32 dimensions
b = 1 + ceil(log2(s+1))           # sign plus magnitude bits per coordinate
packet_bytes = 16 + 4*rnk + ceil(d*b/8)
raw_fp32_bytes = 4*d
```

At levels=127, b=8. A flat 4,096-value float32 tensor takes **4,116 bytes**,
including 20 bytes of metadata, versus 16,384 raw bytes. The formula was checked
against the actual encoder with a small deterministic vector; it is not a
training or speed benchmark. Small tensors can expand because headers are not
free. Zero entries still occupy dense code bits.

Each model tensor gets its own norm and packet, so total codec bytes are the
sum of tensor packets. Use each tensor's dimension for a quantizer bound rather
than silently treating the entire model as one normalization block. The adapter
uses the shared generator in a recorded seed/client/round scheme, and records
quantization error separately from LLZ's extra reconstruction error.

`levels=127` here matches original QSGD s=127, but the IEEE LLZ paper counts
`s-1` intervals, so its s=128 matches our levels=127. Keep that notation
conversion visible in cross-paper configurations.

Complete Flower serialization also includes tensor names, records, routing,
configuration and diagnostics. Our ledger separately counts uploads, downloads
and evaluation exchanges. Neither packet bytes nor serialized object bytes are
physical network traffic. Changing a codec formula cannot retrospectively
change the denominator of a frozen experiment.

## 5. Thesis implications and next handoff

- Hold model, data, local work, participation and evaluation rule fixed when
  isolating compression. Otherwise the experiment also changes optimization.
- Describe QSGD as stochastic quantization with our project-defined dense
  encoding. Report LLZ's incremental saving relative to that explicit baseline.
- Keep distortion relative to the original update separate from *additional*
  LLZ error relative to decoded QSGD. p=0 replay checks the second stage.
- Cite conditional stationarity results with their assumptions. Validation
  accuracy and a passing codec check do not demonstrate convergence.
- Use this note in the quantization/foundations sections of the PDF. Adaptive
  literature and the synthesis are now in [Blocks 3-5](progress.md).

## 6. Source record and checked locations

All source checks were performed on 29 October. Bibliography keys are in
[references.bib](../references.bib); no new entries were needed.

| ID | Primary source / version | Checked passages | Key |
|---|---|---|---|
| Q | Alistarh et al., NeurIPS 2017, [official PDF](https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf) | Section 3.1; PDF pp. 5-6; Eq. (4), Lemma 3.1, coding and Theorem 3.2 | `alistarh2017qsgd` |
| S | Bernstein et al., ICML 2018, [official PDF](https://proceedings.mlr.press/v80/bernstein18a/bernstein18a.pdf) | Sections 3-4; PDF pp. 3-4, 6; Assumptions 1-3 and Theorems 1-2 | `bernstein2018signsgd` |
| F | Reisizadeh et al., AISTATS 2020, [official PDF](https://proceedings.mlr.press/v108/reisizadeh20a/reisizadeh20a.pdf) | Section 2, PDF p. 3; Sections 3-4, PDF pp. 4-7; Algorithm 1, Eq. (6), Assumptions 1-4 and theorem statements | `reisizadeh2020fedpaq` |

Source PDFs, fingerprints and extracted text are local ignored artifacts in
`outputs/literature-blocks-20261029/`; these are reading records, not review
text for redistribution. The source manifest records exact URLs and SHA256.
The signSGD majority-vote page was rendered to verify formulas where text
extraction lost glyphs. No independent proof audit or numerical reproduction
of the cited papers was performed. No access gap blocks this scoped note.
