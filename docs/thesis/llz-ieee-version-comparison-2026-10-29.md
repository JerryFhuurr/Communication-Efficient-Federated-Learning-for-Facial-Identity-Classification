# IEEE LLZ paper: version comparison and project mapping

Prepared 29 October 2026 for the supervisor meeting on 10 November 2026.
This is a source-reading and planning record. No training or implementation
change is part of this task.

## 1. Reference versions

The current literature reference is Yue Cao, Kai Liang, Shuoyao Wang,
Huiru Zhong, and Youlong Wu, *Federated Learning with Lossy Lempel-Ziv
Gradient Compression*, IEEE Transactions on Communications, Early Access,
published 21 October 2026, DOI [10.1109/TCOMM.2026.3745158](https://doi.org/10.1109/TCOMM.2026.3745158).
The publisher's downloaded PDF is an accepted author version, explicitly not
fully edited. It is not the final typeset version of record.

| Source | Identification | Use |
|---|---|---|
| IEEE Early Access | `papers/llz-tcomm-2026-early-access.pdf`, 15 pages; SHA256 `3afda19c845b24316e86d65d8d082a29129930bca6c69216f6a492ea7cd3e0ca` | Current paper, downloaded through DTU Library on 29 October |
| Previously supplied revision | `Paper(1).pdf`, 61-page bundle; clean manuscript at PDF pages 4-18; SHA256 `048076c4ac508efce1c51d2673636c01e7c5a3235767b3e168765f8a7688611a` | Historical implementation/audit source; original remains in the user's OneDrive paper directory |
| Conference predecessor | Zhong, Liang, Wu, Globecom Workshops 2023; DOI `10.1109/GCWkshps58843.2023.10464660` | Historical origin; separate from the journal article |

IEEE page numbers below are both printed and PDF page numbers. For the old
bundle, add three to a clean manuscript page number. Publisher acceptance is
now verified; older dated notes saying it was unverified describe the evidence
available when those notes were written. Keep both bibliography keys to avoid
silently relabelling the source of an earlier audit.

The licensed PDF and its Windows download marker are excluded locally from Git.
The literature review should use our own diagrams and attributed summaries;
redistributing the source PDF is not needed to share our code or review.

## 2. What changed from the manuscript we used

The main paper was read and the key algorithm, theorem and result pages were
rendered for inspection. Normalized text comparisons strip publisher boilerplate,
running page counters, line-break hyphenation and whitespace, and apply Unicode
NFKC. These comparisons are bounded textual evidence, not a formal verification
of the proof, equation typography, or all figure data.

| Component | IEEE location | Comparison with the earlier clean manuscript | Consequence |
|---|---|---|---|
| LLZ-p encoder, Algorithm 2 | p. 6 | Normalized extracted text identical | Existing reconstructed-history implementation interpretation remains applicable |
| Decoder, worked example, quantization convention | pp. 3-6 | Same core steps on reading: explicit following literal, reconstructed dictionary, signed symbols, `s-1` intervals | No codec change identified by this reading; this is not a fresh full code audit |
| Additional-distortion bound, Lemma 1 | p. 9 | Normalized statement and proof text identical | Retain the distinction between quantization error and additional LLZ error |
| Theorem 1, including conditions and bound | p. 9 | Normalized extracted theorem text identical | No new convergence guarantee for our local-epoch/adaptive implementation |
| Error-feedback explanation | pp. 9-10 | The no-EF scope is stated up front; the old separate Remark 4 is removed | Distinguish the theorem's no-EF scope from the empirical default using EF |
| IID, non-IID, sensitivity and EF tables (II-V) | pp. 13-14 | All four normalized tables, including numbers and captions, identical | These experiments were already present in our supplied revision; they are not newly added evidence |
| Framing, references and author material | pp. 1-2, 14-15 | Publication identifiers, editorial wording, reference ordering and biographies differ | Cite DOI/title/year directly; preserve our independently checked bibliography rather than copying reference numbers |

Machine-readable comparison and local extracts:
`outputs/research-preparation-20261029/version-comparison.json`,
`normalized-text-differences.json`, `ieee-pages.txt`, and `revision-pages.txt`.
Page images in that directory cover pp. 6, 9, 11, 13 and 14. A raw text diff
also contains layout-related moves; do not interpret each insertion as new science.

## 3. What the paper establishes, and its boundaries

### Codec and update

Algorithm 1 (p. 3) quantizes a stochastic gradient, applies LLZ to its symbols,
reconstructs at the server, averages gradients, and performs a global SGD step.
Our model-delta pipeline is a different optimization procedure.

LLZ-p searches reconstructed history for a longest approximate match, emits
position/length/following-literal triplets, and decodes using the same history
(pp. 4-6). `p` is an integer-symbol tolerance. At `p=0`, secondary coding adds
no distortion to QSGD; it does not undo QSGD's loss. The notation conversion is
`paper s = project levels + 1`; `levels=127` corresponds to paper `s=128`.
The reconstructed-history prose remains important because some pseudocode
expressions use `g` after initializing a separate reconstructed sequence.

### Theory

Lemma 1 bounds additional dequantized LLZ error by the mismatch count times
`(p * norm / (s-1))^2`. Theorem 1 assumes smoothness/lower boundedness,
unbiased stochastic gradients, bounded gradient moments/variance and an unbiased
QSGD operator with relative variance control (pp. 9-10).

Positive-p LLZ may introduce bias. The sublinear-rate discussion is conditional
on the effective distortion terms shrinking at an appropriate rate; it does
not say every fixed positive tolerance preserves convergence. LLZ-SI is an
empirical extension outside that theorem. Neither the theorem nor lower
validation loss proves convergence of our adaptive local-epoch FedAvg system.
A full independent proof audit is outside this source comparison.

### Experiments and interpretation

The experiments use CIFAR-10/ResNet-18, CIFAR-10/ViT and
CIFAR-100/ResNet-50. Section VI specifies SGD, batch size 64, a decaying
CIFAR-10 learning rate starting at 0.01, even IID partitions and balanced
Dirichlet label skew with alpha in {0.1, 0.3}. EF is used unless stated
otherwise (pp. 11-14). Table IV studies PowerSGD-LLZ-p, not QSGD-LLZ-p.

The paper defines `rho = transmitted compressor bits / original FP32 gradient
bits`, summed across layers and including compressor metadata. Our total logical
Flower-message bytes also include exchanges such as downloads and validation;
these quantities must have separate labels. A reported rho is a retained fraction,
whereas a saving is `1-rho` against the same denominator. The paper's headline
saving relative to QSGD and its FP32-normalized ratios are not interchangeable.
Do not combine a figure's operating point with a table from another experiment.

Examples of paper-reported values, not new project measurements:

| Evidence | Setting | Paper-reported result | Reading implication |
|---|---|---|---|
| Table IV, p. 13 | PowerSGD-LLZ-p, C10/R18; p=0/1/2 | Accuracy 93.61/93.67/93.43%; rho 0.0350/0.0244/0.0183 | More symbol tolerance trades bytes against learning; no universal best p follows |
| Table IV, p. 13 | Window 32/64/128/256 | Accuracy 93.30/93.66/93.22/93.16%; rho 0.0269/0.0255/0.0243/0.0232 | Larger windows need not yield the best accuracy; time cost also matters |
| Table V, p. 14 | QSGD-LLZ-p, C10/R18, EF on/off | Best accuracy 93.75/90.39%; rho 0.0205/0.0043 | EF is a major experimental factor; the realized byte ratio changes too |
| Table III, p. 13 | C100/R50, alpha=0.3 | PowerSGD-LLZ-p 70.01% vs PowerSGD 71.60% | The paper itself contains a case without a hybrid accuracy advantage |

Tables use different accuracy summaries: Table III says final test accuracy;
Table V says best accuracy. Our checkpoint is selected by minimum validation
loss with held-out test unused. Do not pool them as one benchmark ranking.
Section VII (p. 14) lists adaptive dictionary/distortion selection as future
work. This motivates our direction without establishing novelty or priority.

## 4. Paper-to-project comparison

The project column uses the existing frozen adaptive pilot and actual code,
not the generic defaults in `pyproject.toml`. The pilot remains historical
Windows evidence; migration validation did not repeat the research experiment.

| Dimension | IEEE paper | Existing project | Required action / boundary |
|---|---|---|---|
| Task | Three CIFAR/model tasks | DigiFace 100-way identity classification | Keep DigiFace for the current adaptive study; call it an adaptation |
| Model training | Paper architectures and training schedules | Cached ResNet-18 layer-3 features; train layer4 and classifier | Different trainable dimensions, initialization and update distribution |
| FL update | Algorithm 1 stochastic-gradient aggregation | Four-client local-epoch FedAvg model deltas | Do not inherit the theorem automatically |
| Duration/batch | Long epoch/iteration curves; batch 64 | 15 rounds, one local epoch, batch 16 | No numerical accuracy comparison across these protocols |
| EF | Enabled by default empirically; explicit ablation | Absent from QSGD/LLZ/adaptive pilot; present in earlier Top-k baseline | Freeze feedback policy; EF/no-EF is a useful prospective ablation |
| LLZ | Fixed p; LLZ-SI also evaluated | Fixed p=0/1 and adaptive choices p in {0,1,2}; window 128 | Adaptive selection implemented; LLZ-SI not implemented |
| Low-rank hybrids | PowerSGD-LLZ-p/SI versus PowerSGD/DGC | No PowerSGD or DGC implementation | Do not describe a QSGD sensitivity run as reproducing Table IV |
| Adaptive budget | No deployed adaptive p-budget rule; future work | Minimize complete reply bytes subject to measured extra LLZ squared error <= tau * update squared norm | Existing selector works; scientific budget choice remains open |
| Evidence | Paper tasks, seeds/statistics not fully specified in every table | Three-seed fixed-method study; separate one-seed adaptive pilot | Label each sample size and protocol separately |
| Communication | Compressor bits / FP32 gradient bits | Codec packets and serialized logical Flower objects | Report both boundaries clearly; no measured physical network traffic |
| Quality metric | Test accuracy with table-specific summaries | Validation-loss-selected checkpoint; held-out test unused | Preserve test split; avoid best-test selection in our protocol |

Implementation references: [adaptive codec](../../compression/adaptive_llz.py),
[Flower update adapter](../../federated_compression/updates.py),
[pilot runner](../../flower_face/adaptive_study.py),
[frozen protocol](../../outputs/adaptive-study-20261021T104244874038Z/protocol.json),
[pilot report](../../outputs/adaptive-study-20261021T104244874038Z/report.md),
and [earlier paper-to-code audit](llz-paper-audit.md).

## 5. Bounded experimental targets for the next protocol

These are proposals for 4-6 November, not launched runs or a frozen design.
No new dataset, optimizer or model tuning is needed for today's deliverables.

1. **Primary: budget justification and confirmation.** Retain source-matched
   QSGD and p=0 controls, fixed p=1, and one independently specified adaptive
   rule. Freeze a small training-only calibration procedure, validation-based
   selection criterion, confirmation seed(s), byte boundaries and compute cap
   before runs. A 1 pp accuracy tolerance was previously proposed, not approved
   by a theorem or frozen here. The 5.9/7.0 budgets remain historical heuristics.
2. **First paper-inspired ablation: EF on/off (Table V).** Prefer a matched
   QSGD-LLZ-p pair if implementation and CPU budget permit after the PDF is
   shared. It directly addresses a major methodological difference. Audit EF
   state/reset/persistence and count its setup costs before interpreting results.
   A DigiFace run is a methodological adaptation, not a reproduction of CIFAR
   values. EF is an extension to the current LLZ implementation, not completed work.
3. **Alternative smaller target: p/window sensitivity (Table IV).** Reuse
   training-only saved updates for a bounded codec study, or later hold one
   parameter fixed and vary the other in learning runs. Label a QSGD/DigiFace
   study as inspired by Table IV's PowerSGD/CIFAR design. Saved-update bytes
   alone cannot predict accuracy on a changed training trajectory.

Exact numerical reproduction would require the paper's dataset/model,
quantization and symbol-mapping settings, ranks/Top-k budgets, client/local-step
schedule, EF and momentum semantics, initialization/seeds and evaluation rule.
Several operating-point details are not completely specified in the main
experimental description. Record these gaps rather than inventing defaults.
Do not run the full PowerSGD/DGC/ViT/non-IID/client-count matrix before this
meeting. Keep those as review coverage and later research choices.

## 6. Research question and immediate decisions

**Working question:** Can a client select second-stage LLZ tolerance from a
small candidate set to reduce complete upload bytes while controlling measured
additional reconstruction error and meeting an independently evaluated learning
criterion, at an acceptable computational cost?

A local reconstruction constraint is an implementable codec guarantee. Learning
preservation is a separate empirical/theoretical requirement. The paper does
not supply a universal tau, and its p=1 finding is not a reason to set tau=1.
Explain units, calibration data, objective and fallback before showing results.

Decisions before new training: freeze the acceptance criterion; choose the
small candidate-policy set; separate selection and confirmation evidence;
confirm memory/resource allocation under WSL's 8 GB cap; specify the bounded
paper-inspired experiment. The full convergence proof remains a separate open
research task.
