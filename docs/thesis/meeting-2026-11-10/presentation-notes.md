# Presentation notes — supervisor meeting, 10 November 2026

Matched on 9 November to the current eight-slide `adaptive-llz-meeting-20261110.pptx`. Use slides 1–7 for approximately five minutes. Slide 8 is an appendix for questions. Timing allows short pauses and pointing to the tables.

## Slide 1 — Opening and main result (20 seconds)

Today I’ll give a short update on the literature review and adaptive LLZ experiments. The main result is that we reduced training uploads by about 92 percent. However, the current budget failed our accuracy requirement on one confirmation seed, so the budget rule still needs improvement.

## Slide 2 — Compression methods in the literature (45 seconds)

I organized the literature by what each method sends and what extra costs it introduces. Quantization sends values at lower precision. Sparsification sends selected values together with their coordinate indices, often using error feedback. PowerSGD sends low-rank factors. Learned methods need a decoder, so its distribution and updates also matter.

Adaptive compression already has several precedents. My specific question is whether a client can choose an LLZ tolerance after one fixed QSGD draw to reduce the complete message size while staying within an explicit reconstruction-error budget. The full review discusses the assumptions behind each method.

## Slide 3 — One saved client decision (45 seconds)

This table shows a real decision from seed 43, round five, client one. First, the client trains locally and forms its model delta. It quantizes that delta once using QSGD, then tries LLZ with p equal to zero, one and two on the same quantized output.

The extra-error ratios are zero, 5.851 and 6.494. With a budget of 5.9, only p zero and p one are feasible. Although p two gives the smallest message, it exceeds the budget. The client therefore chooses p one and sends 208,554 bytes. The saved observed message size matches this selection.

## Slide 4 — Where 5.9 and 7 came from (40 seconds)

The original budget values came from eight saved training-only updates: four clients at two points in separate local training trajectories. For p one, the lower quartile of the error ratios was about 5.924. Rounding down gave 5.9. The loose rule used the upper quartile for p two and rounded up to seven.

These rules explain the numbers’ origin, but they do not establish learning safety. I assessed learning separately using our agreed criterion: adaptive accuracy should fall by no more than one percentage point compared with matched QSGD.

## Slide 5 — Budget sensitivity on development seed 42 (50 seconds)

I tested the three budgets using the same DigiFace setup: 100 identities, four IID clients, 15 rounds and one local epoch. The model trains ResNet-18 layer four and its classifier using cached layer-three features.

QSGD reached 83.5 percent validation accuracy, so the development threshold was 82.5 percent. At a budget of 5.9, adaptive LLZ reached 83.2 percent and sent about 40.9 megabytes of uploads. Increasing the budget to 6.5 reduced uploads further, but accuracy fell to 81.1 percent. At seven, it fell to 78.9 percent.

Only 5.9 passed among these tested budgets. I then froze that policy before confirmation.

## Slide 6 — Frozen confirmation on seeds 43 and 44 (60 seconds)

For confirmation, I ran matched QSGD and adaptive experiments on seeds 43 and 44, keeping the code, data and training settings fixed. Each pair used the same initialization, and I did not change the budget after seeing the results.

On seed 43, QSGD reached 83.6 percent and adaptive reached 83.3 percent. The drop was 0.3 percentage points, so it passed. On seed 44, the accuracies were 83.2 and 82.0 percent. That 1.2-point drop failed the one-point requirement.

Both adaptive runs reduced uploads by about 92 percent and passed the selection audit. This shows that the selector follows its rule, while the learning outcome still varies. These are validation-selected checkpoints, and the held-out test remains unused.

## Slide 7 — Interpretation and next research decision (40 seconds)

We now have a working adaptive LLZ implementation and a reproducible path from individual client decisions to the final results. The remaining issue is how to choose a budget that preserves learning more consistently.

My proposed next step is to study a tighter budget, or one that changes during training, using fresh development evidence. We could also consider total reconstruction error or update alignment, since the current budget only measures added LLZ error. Candidate-search cost also needs attention.

I’d like your advice on which budget approach to prioritize and when to add a matched error-feedback comparison.

## Slide 8 — Appendix: communication boundaries and controls (only if asked)

This table provides the seed-42 controls. LLZ with p zero reproduced QSGD’s numerical training results because it adds no further loss to the quantized representation. Fixed p one sent fewer bytes but reached lower validation accuracy in this setup.

The large saving applies to training uploads. Total serialized-object savings were about 10.2 percent because downloads and other exchanges account for most of the total. These are logical Flower object sizes. We did not capture physical network traffic.

Adaptive runs also took longer: around 46 and 56 minutes, compared with roughly 12 minutes for QSGD. Those are complete launch times, so they include training and evaluation as well as compression.

## Presenter cues

- On slide 3, point to the p=2 row before explaining why p=1 wins.
- On slide 5, point to the red 82.5% gate, then the accuracy curve and upload table.
- On slide 6, state the seed-44 failure clearly. Do not retell it as a pass based on average accuracy.
- Finish the main talk after slide 7. Use slide 8 when the discussion turns to p=0, communication accounting or runtime.
- If asked for the live walkthrough, run `python3 demo_saved_decision.py` from this folder. It rechecks saved diagnostics and does not retrain or regenerate the original update.
- The literature PDF in this folder is the dated 2 November snapshot. The slides contain the newer budget-sweep and confirmation results.

## Questions to ask

1. Should I first investigate a tighter budget schedule, or change the quality measure to total error or update alignment?
2. Should the next matched comparison introduce error feedback before adding more compression methods?
3. Which parts of the literature review need strengthening for the thesis chapter?

## Evidence

The PowerPoint supplies the slide order, displayed values and chart series. Supporting data are `confirmation-summary.json`, `budget-sweep.csv` and `saved-decision.json` in this folder. The frozen source identity is in `source-sha256.txt`. The literature review is available at https://www.overleaf.com/project/6aeb378d04c7d5dba6f6b8f9 .
