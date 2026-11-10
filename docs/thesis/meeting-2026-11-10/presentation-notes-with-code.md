# Speaker notes with code walkthrough

Slides 1–7 form the main talk, approximately five minutes. Slides 8–9 are backup. Code is editable text, excerpted from the project with formatting adjusted and unrelated lines omitted. No new training is needed.

## Slide 1 — Opening: the questions from last time (20 seconds)

*Cue: Introduce the purpose, then move on. The headline stays visible.*

Today I’ll focus on the questions from our last meeting: where the budget values came from, how I obtained the results, and how this fits the literature. The main finding is about 92 percent lower training uploads, but the accuracy requirement failed on one confirmation seed.

## Slide 2 — Literature review (35 seconds)

*Cue: Point to the table. Summarize it rather than reading every row.*

I prepared the literature review as a PDF with comparison tables and figures. This table summarizes the main research lines.

Their costs differ. For example, Top-k needs coordinate indices, low-rank methods send factors, and learned methods may need decoder setup or refresh.

Adaptive compression already has precedents. My specific question is whether choosing the LLZ tolerance after one QSGD draw can reduce the complete message size while meeting an explicit error budget.

## Slide 3 — How one client gets its result (35 seconds)

*Cue: Point to p=2, then p=1. The next slide shows the actual selection code.*

This is a real decision from seed 43. Each client tries p zero, one and two on the same QSGD output.

Here, p two gives the smallest message, but its extra-error ratio exceeds the budget. The client therefore sends p one, at around 200 kilobytes. The saved message size matches that selection.

The next slide shows the code behind this decision.

## Slide 4 — Code behind the client decision (35 seconds)

*Cue: Point to feasible, then min. Explain the purpose of each block without reading the code aloud.*

The first block keeps candidates whose added LLZ error, divided by the original update’s squared norm, is within the budget. The admissible function also handles a zero update safely.

The second block chooses the feasible candidate with the lowest complete message cost. If two messages have the same cost, it prefers the lower p.

This is the project’s production selector. The saved example on the previous slide shows one outcome of this rule.

## Slide 5 — Why the budgets were 5.9 and 7 (65 seconds)

*Cue: Take time here. Distinguish the origin of a number from evidence that it preserves accuracy.*

The starting values came from eight saved training-only updates: four clients at two points in their local training.

For p one, the lower quartile of the error ratios was about 5.92. Rounding down gave 5.9.

For the looser setting, the rule took the larger of the upper quartile for p two, about 6.93, and 1.1 times the tighter budget. The first value was larger, so rounding up gave seven.

This explains the numbers’ origin. It does not show that they preserve learning, especially with only eight snapshots. I therefore checked accuracy separately, using a one-percentage-point margin set before confirmation.

Also, 5.9 is a relative squared-error limit. It does not mean 5.9 percent, and p and the budget are different parameters.

## Slide 6 — How the budget behaved during training (60 seconds)

*Cue: Point to the quality threshold, then the accuracy curve. Round upload sizes when speaking.*

I then tested 5.9, 6.5 and seven on development seed 42, keeping the model, data and training settings fixed. These were full training runs.

After each round, I measured validation loss and accuracy. The checkpoint with the lowest validation loss supplies the reported accuracy. Upload totals add the serialized training replies across all rounds.

QSGD reached 83.5 percent, so the one-point margin put the threshold at 82.5. At 5.9, adaptive LLZ reached 83.2 percent on about 40 megabytes of uploads.

Larger budgets sent less data, but accuracy fell below the threshold. Only 5.9 passed among these three tested values, so I froze it before trying the confirmation seeds.

## Slide 7 — What the confirmation showed (45 seconds)

*Cue: Point to PASS and FAIL. End the main talk here and invite discussion.*

For confirmation, I used seeds 43 and 44 with matched QSGD and adaptive runs. I kept the budget fixed.

Seed 43 lost 0.3 percentage points and passed. Seed 44 lost 1.2 points and failed the one-point requirement. Both reduced training uploads by about 92 percent.

The selection audit found no errors. So the client decisions followed the declared rule, but that rule did not meet the accuracy requirement consistently. These are validation-selected results, and the test set remains unused.

That is the main result I wanted to bring back to you. I’d like your view on how to interpret the trade-off.

## Slide 8 — Backup: controls, total communication and runtime (backup)

*Cue: Use only if asked. Stop after the relevant paragraph rather than reading all of it.*

For p zero, LLZ adds no further loss to QSGD, so it reproduces the same numerical training results. Fixed p one compresses more but reached lower accuracy in this setup.

The 92 percent saving compares training uploads with QSGD. Lossless LLZ already gives much of that saving. On the development seed, adaptation saves about another third of uploads compared with p zero.

When I include downloads and other exchanges, total serialized savings are around ten percent. We measured logical Flower object sizes rather than captured network traffic.

Adaptive also took longer: roughly 45 to 55 minutes, compared with about twelve for QSGD. Those are whole-run timings, including training and evaluation.

## Slide 9 — Backup: code behind the reported results (backup)

*Cue: Show only if asked where final accuracy or byte totals come from.*

This code reads the saved checkpoint record and communication totals for each completed run. The checkpoint was selected by the lowest validation loss.

The reported accuracy comes from that checkpoint, while training uploads come from the train-uplink serialized-object total. Those values feed the comparison summary and slide tables.

The full script also checks source and data fingerprints and audits the saved client decisions.

## What to point to on the code slides

Slide 4: `feasible` applies the reconstruction-error rule, then `min` chooses the smallest complete reply. The full production code separately checks the p=0 fallback. Slide 9: `best_checkpoint.json` supplies validation accuracy at the minimum-loss checkpoint; `communication.json` supplies aggregate training uploads. The full summary script additionally checks run completeness, source/data fingerprints, initialization and candidate diagnostics.

## Source excerpts

feasible = [c for c in candidates if admissible(
    c.stats['llz_reconstruction_squared_error'],
    common['qsgd_input_squared_norm'], budget)]

chosen = min(feasible,
             key=lambda c: (costs[c.p], c.p))

cp = json.loads((path / "best_checkpoint.json").read_text())
comm = json.loads(
    (path / "communication.json").read_text())["totals"]

record = {
    "accuracy_percent": cp["validation_accuracy"] * 100,
    "training_upload_bytes":
        comm["train"]["uplink"]["serialized_object_bytes"],
}
