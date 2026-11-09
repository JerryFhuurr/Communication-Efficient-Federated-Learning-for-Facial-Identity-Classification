# Five-minute presentation notes — 10 November 2026

Prepared 5 November using completed WSL results. Slides 1–7 total 300 seconds. Slides 8–9 are supporting material. Pause at slide 3 to show the saved example only if useful; use its printed output instead of adding time to the talk.

## Slide 1 — 20 seconds

Today I will show the literature review, explain how I selected the LLZ error budget, and trace the experimental results. The main finding is that adaptive LLZ cuts training uploads by about 92 percent, but the frozen budget does not preserve accuracy within our chosen margin on both confirmation seeds.

## Slide 2 — 45 seconds

The review covers six main lines of compression. Quantization changes numerical precision. Sparsification sends coordinates and their indices, often with error feedback. PowerSGD sends low-rank factors. Learned coding needs a compatible decoder whose transfer costs must be counted. Adaptive precision and side-information methods already exist. LLZ adds dictionary coding after quantization. Our narrower research question is whether choosing the LLZ tolerance after one fixed QSGD draw gives a useful learning and communication trade-off.

## Slide 3 — 45 seconds

Here is one real decision from the seed-43 confirmation run. The client forms its local model delta and makes one QSGD draw. It then encodes and locally decodes three LLZ tolerances. The extra squared error is normalized by the original update squared norm. Here p equals two gives the smallest message, but its error ratio is 6.494, above the 5.9 budget. The selector therefore sends p equals one at 208,554 bytes. The observed message size matches that choice. The supplied demonstration command prints these steps directly from the saved record.

## Slide 4 — 40 seconds

The numbers were not prescribed by the paper. They came from eight saved training-only updates. I rounded the lower quartile of the p-equals-one error ratios down to 5.9. The loose rule used the upper quartile for p equals two and rounded up to seven. This makes the origin reproducible, but eight clustered snapshots are weak calibration evidence. I therefore assessed learning separately, using the agreed one-percentage-point accuracy margin. Meeting a reconstruction limit is only a check that the selector follows its rule.

## Slide 5 — 50 seconds

I then tested 5.9, 6.5 and seven using the same seed-42 WSL setup. QSGD reached 83.5 percent, which puts the development threshold at 82.5 percent. Adaptive 5.9 reached 83.2 percent and used 40.924 megabytes of uploads. At 6.5 the accuracy fell to 81.1 percent, and at seven it fell to 78.9 percent. Both failed the margin despite sending less data. Under the declared rule, 5.9 was the only passing tested budget. I froze it before the confirmation runs. This supports a development choice among these points, rather than a universal optimum.

## Slide 6 — 60 seconds

The confirmation is less positive than the first seed suggested. On seed 43, QSGD reached 83.6 percent and adaptive reached 83.3, so it passed. On seed 44, the corresponding values were 83.2 and 82.0, a 1.2-point drop, so it failed the frozen gate. Uploads fell by about 92 percent in both pairs. All selections passed the saved diagnostic audit, which shows correct policy execution but does not establish learning safety. I excluded an incomplete seed-44 attempt and used its completed retry with exactly the same settings. I have kept the failure and have not tuned the budget using these confirmation results.

## Slide 7 — 40 seconds

My conclusion is that we now have a reproducible adaptive LLZ implementation and a measurable communication gain, but the current calibration rule does not pass our accuracy requirement consistently. I propose using fresh development evidence to study a tighter budget or one that changes during training. A second option is to account for total error or alignment with the original update, since the current constraint only measures added LLZ error. I also want to reduce candidate-search cost. I would like your advice on the accuracy margin and whether a matched error-feedback ablation should follow this budget work.

## Slide 8 — appendix

Use only if asked. These are WSL controls, not the older Windows pilot. LLZ p=0 adds no loss to the quantized reconstruction. It still follows a lossy QSGD first stage. Whole-launch times include training, evaluation, codecs and startup, and they are descriptive rather than randomized runtime benchmarks. Total traffic is dominated by downloads and other exchanges. The existing Top-k baseline includes error feedback while this LLZ study does not, so it is not a matched comparator here.

## Slide 9 — appendix

Use the meeting README for the exact paths and commands. The demonstration recomputes feasibility and minimum-byte selection from saved diagnostics. It does not rerun quantization or regenerate the original model delta. Full training reproduction requires the local data/cache and recorded environment. Git HEAD alone is insufficient because the frozen run used a working tree with changes. The source ZIP includes the matching Python/TOML snapshot and an explicit fingerprint manifest.

## Likely questions and short answers

**Why exactly 5.9 and 7?** The values came from stated rounding rules applied to training-only candidate-error quantiles: Q25(p=1)=5.923659 gave 5.9; max(Q75(p=2)=6.928620,1.1×5.9) rounded up gave 7. These are traceable pilot choices. Seed-42 development supported 5.9 among the tested budgets, but confirmation failed on seed 44, so the evidence does not establish a generally safe threshold.

**What does tau mean?** It limits added LLZ squared error relative to the original model-delta squared norm. Tau=5.9 is dimensionless and allows an extra-error norm up to sqrt(5.9), about 2.43 times the update norm. It is neither a percentage nor an accuracy-loss allowance. It controls added LLZ error rather than total reconstruction error.

**Does max rho≈5.895 prove the budget is appropriate?** It shows the chosen update met the configured constraint. The selector excludes values above tau by construction. Learning suitability requires the separate matched training comparison, which failed its per-seed accuracy gate on seed 44.

**Can we choose a new threshold using seed 44?** That would make the confirmation outcome part of policy development. Keep this failure in the report and use a newly declared development/confirmation protocol for any revised rule.

**Why not simply use p=1 as in the paper?** Its empirical success depends on the task and training setup. Our WSL seed-42 p=1 control reached 79.3% versus 83.5% for QSGD. Our local-epoch face task differs from the paper's gradient experiments. This motivates adaptation without contradicting their reported cases.

**Is p=0 lossless?** LLZ p=0 exactly preserves the quantized reconstruction. The QSGD first stage still introduces error relative to the original full-precision delta. The seed-42 p=0 learning records numerically replay QSGD.

**Why did a correct selector lose accuracy?** The local budget only constrains extra reconstruction error. It does not constrain accumulated optimization error, alignment with the update, or validation accuracy across client/round trajectories. Error feedback is also disabled in this study.

**How much communication did we save?** Training upload objects fell about 92.1% and 92.2% on seeds 43/44. Total serialized objects fell about 10.2%, because downloads and other exchanges dominate. These are logical Flower sizes, not physical network measurements. MB means one million bytes.

**What is the runtime cost?** QSGD launches took 11.6 and 12.4 minutes; adaptive took 46.3 and 56.1 minutes. These include the full launch and are descriptive, not randomized codec timings. Multiple candidate encodings contribute overhead.

**Why not include Top-k here?** The previous Top-k baseline used error feedback while this LLZ study did not. These runs isolate the adaptive budget. A fair future comparison should match residual rules, training conditions and complete byte accounting.

**What is novel given existing adaptive compression?** Adaptation is already established. The narrow implementation is measured-byte minimization over locally decoded LLZ tolerances after one shared QSGD draw, under an explicit additional-error constraint. AdaQuantFL/DAdaQuant adapt precision; KLMS uses distributional side information; ASCEND selects among compressors using reward feedback; discrepancy-aware calibration scores components by model-output discrepancy. The review discusses these precedents. Do not claim priority or a new theorem.

**Are the results test accuracy or statistical equivalence?** They are accuracies at checkpoints selected by minimum validation loss. The held-out test remains unused. Two confirmation seeds are limited evidence; the per-seed criterion failed and no equivalence claim is justified.

**What happened to seed 44?** The first adaptive attempt was incomplete. Its same-settings retry completed all 15 rounds. Only that completed retry appears in the comparison, with the earlier attempt retained for provenance.

**Can you show exactly where a result comes from?** Open the saved row, divide each candidate squared error by `qsgd_input_squared_norm`, apply tau, select the cheapest feasible complete message and compare observed bytes. Then show `best_checkpoint.json`, `communication.json` and `confirmation-summary.json` for the full run. Code and manifest fingerprints identify the matched protocol.

## Questions to ask the supervisor

1. Should the next budget use total error or update alignment, or should we first study a tighter schedule over training rounds?
2. Does the 1.0 percentage-point accuracy margin remain suitable for the thesis objective? Any revised margin should be declared prospectively.
3. After the budget revision, should the next comparison prioritize matched error feedback or a matched Top-k baseline?
4. Which claims or missing literature links need the most work before this review becomes a thesis chapter?

## Short opening if the meeting runs late

“I finished the literature review and made the adaptive LLZ study reproducible. The budget values now have a documented calibration origin and a matched sensitivity study. The frozen 5.9 policy reduced uploads by about 92 percent, but seed 44 lost 1.2 accuracy points and failed our one-point gate. I would like to discuss how to improve the budget rule using fresh development evidence.”
