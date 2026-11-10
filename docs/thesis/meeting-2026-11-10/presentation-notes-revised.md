# Speaker notes — 10 November meeting

Slides 1–8: about five minutes including pauses. Slides 9–10: backup only. No next-steps slide.

## Slide 1 — Motivation and setting (40 seconds)

*Cue: Introduce the system briefly, then the question. Do not read the setup line by line.*

In federated learning, clients train locally and exchange model updates with a server. Repeating those exchanges creates the communication cost we want to reduce.

I use Flower to simulate four clients on a small DigiFace synthetic-face classification task. This pilot is a controlled initial study. QSGD, fixed LLZ and the adaptive selector are implemented and tested.

The question is whether adapting LLZ can save communication while keeping learning quality. Device heterogeneity is outside this study.

Evidence (not spoken): outputs/adaptive-dev-seed42-wsl-20261102/summary.json; outputs/adaptive-budget-confirmation-20261103/confirmation-summary.json

## Slide 2 — Comparison with the closest rate-constrained alternative (35 seconds)

*Cue: Explain the objective row, then point to the different distortion and cost definitions.*

RC-FED is a close comparison: it designs a scalar quantizer to reduce expected distortion under a coded-rate constraint. Here, QSGD stays fixed; I select an LLZ tolerance using measured reply bytes and an added-error limit.

So adaptation itself is not the contribution claim. The narrower question concerns this LLZ selection rule, its learning reliability and its complete cost. I have not run a matched RC-FED benchmark.

Evidence (not spoken): Hamidi & Bereyhi, arXiv:2409.06319v1 §§2–4: https://arxiv.org/html/2409.06319v1 ; published DOI: https://doi.org/10.1109/ICASSP49660.2025.10889213

## Slide 3 — One saved client decision (35 seconds)

*Cue: Point to p=2, then p=1. This is the step-by-step evidence requested last time.*

Here is one saved client decision. The client forms its local model delta, applies QSGD once, and tries p zero, one and two on that same output.

Each candidate is decoded locally. I measure the extra LLZ error relative to the original update’s squared norm. Here p two is smaller but exceeds the budget, so the client sends p one, at around 200 kilobytes. The recorded message matches that choice.

Evidence (not spoken): saved-decision.json; compression/adaptive_llz.py:76–116

## Slide 4 — Code behind the client decision (20 seconds)

*Cue: Point to feasible and min; explain the actions without reading Python aloud.*

This is the production selection code. First it keeps candidates that meet the error budget. Then it chooses the feasible candidate with the smallest complete reply cost. Byte ties prefer a lower p.

The full implementation also checks that p zero remains an exact fallback to the QSGD reconstruction.

Evidence (not spoken): compression/adaptive_llz.py:44–46,107–116; reformatted excerpt, fallback guard omitted

## Slide 5 — Why the budgets were 5.9 and 7 (45 seconds)

*Cue: Separate the origin of the numbers from evidence that they preserve learning.*

The budgets came from eight training-only snapshots, rather than from a theorem. The lower quartile for p one was about 5.92; rounding down gave 5.9. The upper quartile for p two was about 6.93; the loose rule rounded that up to seven.

These quartiles were exploratory calibration choices. Eight snapshots are weak evidence, so I assessed learning separately using a one-percentage-point accuracy margin fixed before confirmation.

Also, 5.9 is a relative squared-error limit, not 5.9 percent.

Evidence (not spoken): docs/thesis/adaptive-budget-sensitivity-2026-11-03.md; outputs/adaptive-study-20261021T104244874038Z/study.json; docs/thesis/meeting-2026-11-10/confirmation-protocol.md

## Slide 6 — How the budget behaved during training (40 seconds)

*Cue: Point to the threshold and curve. Explain which saved records supply accuracy and bytes.*

I ran full training at 5.9, 6.5 and seven on development seed 42, with the other settings fixed. Accuracy comes from the lowest-validation-loss checkpoint; uploads sum all serialized training replies.

QSGD reached 83.5 percent, so the one-point margin gave an 82.5 threshold. At 5.9, adaptive LLZ reached 83.2 percent with about 40 megabytes of uploads. Larger budgets sent less but failed the threshold. Only 5.9 passed among these tested values, so I froze it.

Evidence (not spoken): outputs/adaptive-budget-sweep-20261103/full-run-summary.csv; docs/thesis/adaptive-budget-sensitivity-2026-11-03.md

## Slide 7 — What the confirmation showed (35 seconds)

*Cue: Keep the failed seed visible. Do not describe this as a code failure or established statistical equivalence.*

With the budget frozen, seed 43 lost 0.3 percentage points and passed. Seed 44 lost 1.2 points and failed. Both reduced training uploads by about 92 percent.

The saved selection audit found no errors. That means the policy ran as specified, but its reconstruction constraint did not preserve learning within our chosen margin consistently. These are validation-selected results; the test set remains unused.

Evidence (not spoken): outputs/adaptive-budget-confirmation-20261103/confirmation-summary.json; docs/thesis/meeting-2026-11-10/confirmation-protocol.md

## Slide 8 — Critical interpretation and practical value (40 seconds)

*Cue: Compare adaptive with p=0, then explain total bytes and runtime. End here; there is no next-steps slide.*

Lossless LLZ already provides much of the saving. Compared with p zero on the development seed, adaptation saves around another third of uploads, but less than half a percent of total serialized bytes. Fixed p one sends less again, with lower accuracy here.

Adaptive confirmation runs also took roughly four times as long as QSGD. Those are whole-run timings.

So the implementation provides a reproducible trade-off, but a reliable learning benefit and a net practical advantage over lossless LLZ remain unproven.

Evidence (not spoken): outputs/adaptive-dev-seed42-wsl-20261102/summary.json; outputs/adaptive-budget-confirmation-20261103/confirmation-summary.json

## Slide 9 — Backup: code behind the reported results (backup)

*Cue: Use only if asked where checkpoint accuracy and communication totals come from.*

The script reads best_checkpoint.json for accuracy at the checkpoint with the lowest validation loss. communication.json supplies training-uplink serialized-object totals across the full run.

The full summary script also checks completeness, source and data fingerprints, initialization and the saved client selections. The live walkthrough rechecks a saved decision; it does not rerun training.

Evidence (not spoken): experiments/summarize_adaptive_budget_confirmation.py:20–41,85–103; demo_saved_decision.py

## Slide 10 — Backup: literature, assumptions and review status (backup)

*Cue: Use the relevant paragraph only. Explain the review revision candidly if asked.*

The review maps quantization, sparsification and feedback, low rank, learned coding, adaptive allocation and LLZ. Their costs motivate measuring actual messages: indices, factors, shared state and decoder transfers are not free. Our project-defined dense code means our own packed representation of QSGD symbols and metadata, rather than a mandated QSGD bitstream.

Conditionally unbiased means that, for a fixed input and training history, averaging over the compressor’s fresh randomness recovers that input. A later biased, data-dependent LLZ selection does not automatically retain this property.

EF21 is relevant later error-feedback work using contractive compression. Our no-feedback local-delta protocol and added-LLZ-error budget do not establish its compressor and algorithm assumptions. No EF21 theorem is claimed for this study.

The search was a bounded narrative review: foundational papers, then dated primary-source searches for adaptive, learned and LLZ compression. The 29 October update screened seven recent candidates and inspected three new close works in detail, recording access gaps. It was not a systematic or exhaustive search.

The revised PDF still needs broader learned-code coverage, deeper comparisons and a stronger connected argument. The current AE experiment remains offline, not a demonstrated learned FL compressor.

LLM disclosure: ChatGPT and Codex assisted literature discovery, synthesis and drafting, as well as code and presentation preparation. Primary-source checks and access limits are documented; the comparison remains subject to revision.

Evidence (not spoken): Search: docs/thesis/review-preparation-20261030/04-recent-work-search.md (29 Oct 2026); quantization and learned-method notes in the same directory; EF21: https://proceedings.neurips.cc/paper/2021/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html ; review references: docs/thesis/references.bib

## Questions about terminology

**What is a model delta?** The locally trained model minus the model received from the server. Our compressor acts on that multi-step local delta, rather than directly on a single mini-batch gradient.

**What does the budget control?** Sum of squared additional LLZ reconstruction error, relative to the original local delta’s squared norm. It excludes the QSGD first-stage error; it is not an accuracy guarantee or a percent.

**What remains proposed?** A justified budget with reliable learning performance, evidence against the closest matched alternatives, and a demonstrated benefit after total communication and computational cost.

**Why is seed 44 a failure?** It failed the declared accuracy margin. The evidence does not identify a specific cause; correct selection and bounded added error do not guarantee final validation accuracy.
