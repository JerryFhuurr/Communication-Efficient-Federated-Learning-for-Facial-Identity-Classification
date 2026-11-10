# Speaker notes — supervisor meeting, 10 November 2026

Matched to the focused version of your edited deck. Six main slides, approximately five minutes including pauses. Slide 7 is backup. Read the spoken paragraphs; cues and evidence are for reference.

## Slide 1 — Opening: the questions from last time (20 seconds)

*Cue: Introduce the purpose, then move on. The headline stays visible.*

Today I’ll focus on the questions from our last meeting: where the budget values came from, how I obtained the results, and how this fits the literature. The main finding is about 92 percent lower training uploads, but the accuracy requirement failed on one confirmation seed.

## Slide 2 — Literature review (35 seconds)

*Cue: Point to the table. Summarize it rather than reading every row.*

I prepared the literature review as a PDF with comparison tables and figures. This table summarizes the main research lines.

Their costs differ. For example, Top-k needs coordinate indices, low-rank methods send factors, and learned methods may need decoder setup or refresh.

Adaptive compression already has precedents. My specific question is whether choosing the LLZ tolerance after one QSGD draw can reduce the complete message size while meeting an explicit error budget.

## Slide 3 — How one client gets its result (60 seconds)

*Cue: Point first to the four steps, then to p=2 and p=1 in the table.*

Here is one real client decision. The client trains locally and forms an update. It applies QSGD once, then tries LLZ at p zero, one and two on that same quantized output.

For each candidate, I measure the extra squared error added by LLZ and divide it by the squared magnitude of the original update. That gives the error ratio shown here.

With a budget of 5.9, p one is just inside the limit and p two is outside it. Although p two is smaller, the client sends p one, at around 200 kilobytes. The recorded message size matches that choice.

The server then decodes the chosen messages and combines the client updates using FedAvg.

## Slide 4 — Why the budgets were 5.9 and 7 (65 seconds)

*Cue: Take time here. Distinguish the origin of a number from evidence that it preserves accuracy.*

The starting values came from eight saved training-only updates: four clients at two points in their local training.

For p one, the lower quartile of the error ratios was about 5.92. Rounding down gave 5.9.

For the looser setting, the rule took the larger of the upper quartile for p two, about 6.93, and 1.1 times the tighter budget. The first value was larger, so rounding up gave seven.

This explains the numbers’ origin. It does not show that they preserve learning, especially with only eight snapshots. I therefore checked accuracy separately, using a one-percentage-point margin set before confirmation.

Also, 5.9 is a relative squared-error limit. It does not mean 5.9 percent, and p and the budget are different parameters.

## Slide 5 — How the budget behaved during training (60 seconds)

*Cue: Point to the quality threshold, then the accuracy curve. Round upload sizes when speaking.*

I then tested 5.9, 6.5 and seven on development seed 42, keeping the model, data and training settings fixed. These were full training runs.

After each round, I measured validation loss and accuracy. The checkpoint with the lowest validation loss supplies the reported accuracy. Upload totals add the serialized training replies across all rounds.

QSGD reached 83.5 percent, so the one-point margin put the threshold at 82.5. At 5.9, adaptive LLZ reached 83.2 percent on about 40 megabytes of uploads.

Larger budgets sent less data, but accuracy fell below the threshold. Only 5.9 passed among these three tested values, so I froze it before trying the confirmation seeds.

## Slide 6 — What the confirmation showed (45 seconds)

*Cue: Point to PASS and FAIL. End the main talk here and invite discussion.*

For confirmation, I used seeds 43 and 44 with matched QSGD and adaptive runs. I kept the budget fixed.

Seed 43 lost 0.3 percentage points and passed. Seed 44 lost 1.2 points and failed the one-point requirement. Both reduced training uploads by about 92 percent.

The selection audit found no errors. So the client decisions followed the declared rule, but that rule did not meet the accuracy requirement consistently. These are validation-selected results, and the test set remains unused.

That is the main result I wanted to bring back to you. I’d like your view on how to interpret the trade-off.

## Slide 7 — Backup: controls, total communication and runtime (backup)

*Cue: Use only if asked. Stop after the relevant paragraph rather than reading all of it.*

For p zero, LLZ adds no further loss to QSGD, so it reproduces the same numerical training results. Fixed p one compresses more but reached lower accuracy in this setup.

The 92 percent saving compares training uploads with QSGD. Lossless LLZ already gives much of that saving. On the development seed, adaptation saves about another third of uploads compared with p zero.

When I include downloads and other exchanges, total serialized savings are around ten percent. We measured logical Flower object sizes rather than captured network traffic.

Adaptive also took longer: roughly 45 to 55 minutes, compared with about twelve for QSGD. Those are whole-run timings, including training and evaluation.

## If the supervisor asks for the exact numbers

The slides retain the exact measurements. In speech, use about 40 MB, around 200 KB and roughly 92% unless a precise comparison is needed. Keep the budgets 5.9 / 6.5 / 7 and the accuracy drops 0.3 / 1.2 percentage points precise. The 1-point acceptance rule was declared before confirmation.

## Saved demonstration

From the meeting folder, run `python3 demo_saved_decision.py`. It rechecks the saved candidate diagnostics and final comparison without training. The client-decision example is not itself a full-training reproduction.

## Supporting details

The maximum selected ratio just below 5.9 shows feasibility under the rule, not that the threshold is appropriate for learning. The eight calibration snapshots are clustered observations. Confirmation did not pass on both seeds. The test set remains unused.

The literature PDF in the send package is the dated 2 November snapshot. The slides supply the later experimental results.
