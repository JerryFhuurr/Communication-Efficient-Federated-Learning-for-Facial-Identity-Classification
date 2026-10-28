# Supervisor briefing: current thesis stage

25 September 2026

## Objective

Investigate whether LLZ-p can reduce communication beyond QSGD in federated
learning while preserving useful accuracy. The current demonstration uses
closed-set classification of 100 synthetic DigiFace identities. The contribution
so far is an auditable implementation and empirical study, not a newly invented
compression algorithm.

## Completed work

The project has a modular Flower/PyTorch framework, standalone QSGD and LLZ-p
codecs, a reusable dataset interface, deterministic comparisons, and explicit
packet/message accounting. A stronger model trains ResNet-18's final stage and
classifier over cached frozen features. The paper-to-code audit found no blocking
codec defect under the documented reconstructed-history interpretation; its
focused batch passed 72 tests. That audit is not author certification or a
proof of the FedAvg implementation's convergence.

The written package now contains a focused literature review, methodology,
experimental protocol, implementation audit, results discussion, limitations,
reproducibility requirements, and a 12-entry bibliography. The supplied 2026 LLZ
revision is identified as a manuscript, not a verified accepted publication.

## Main result

In one matched seed-42 comparison with four clients and 15 rounds:

| Method | Selected validation accuracy | Training upload |
|---|---:|---:|
| Uncompressed | 83.3% | 2028.759 MB |
| QSGD | 83.4% | 506.835 MB |
| QSGD + LLZ p=0 | 83.4% | 59.749 MB |
| QSGD + LLZ p=1 | 78.7% | 12.483 MB |

LLZ p=0 replayed QSGD exactly for all rounds and the selected checkpoint. It
reduced QSGD uploads by 88.21% and total serialized bytes by 9.79%. Moving from
p=0 to p=1 saved another 1.15% of total bytes but lost 4.7 accuracy percentage
points. Full-state training and validation downloads dominate the remaining
communication. Accuracy is selected by minimum validation loss; byte totals
cover all 15 rounds and include metadata. They are not network captures.

## Interpretation and limits

This is promising evidence for lossless secondary compression, with p=0 the
current working choice. The one-image accuracy difference between uncompressed
and QSGD is not evidence that compression improves accuracy. This pilot does
not establish statistical equivalence, convergence, performance on unseen
identities, or deployment speedup. The held-out test set remains unused.

The p=1 result cannot directly reproduce or contradict the supplied paper:
our model-delta protocol, dataset, training budget, accounting, and absence of
error feedback differ. Dedicated codec timing, stronger-model seed confirmation,
non-IID experiments, Top-k, error feedback, and LLZ-SI remain future work.

## Decisions requested

1. Is a reproducible empirical characterization of LLZ-p on FedAvg updates an
   adequate central contribution, or is a new algorithmic extension required?
2. Is the current synthetic closed-set task sufficient for the demonstration,
   and what additional task is necessary for the final thesis claims?
3. Should error feedback be the first lossy extension, before LLZ-SI?
4. Are the proposed paired seeds and a 1.0-point practical accuracy-loss margin
   appropriate? The margin is proposed after the pilot, not preregistered.

The immediate implementation priority is encoding/decoding and round timing.
The next bounded training proposal is seeds 43 and 44 for uncompressed, QSGD,
and p=0, subject to source/protocol compatibility. No additional experiments
were launched to prepare this briefing.

Supporting chapters and evidence links: [documentation index](README.md).
