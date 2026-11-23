# CIFAR-10 distortion budget calibration protocol

This note transcribes the rules stored by the local WSL batch before training in that batch. The original timestamp belongs to the machine-generated protocol, rather than to this note. Preserve the original JSON with the run artifacts when sharing the results.

| Record | Value |
| --- | --- |
| Frozen protocol | `outputs/cifar10-adaptive-local-seed42-20261120-retry/protocol.json` |
| Creation timestamp (UTC) | `2026-11-20T08:53:21.269065+00:00` |
| Protocol file SHA-256 | `4a0a5788ab99b607f2ef1d0e9db022728478c7b2f09fd02257f5aeaf62d2ac1e` |
| Frozen training source SHA-256 | `d7d43bbe19c5d3972a9c4562449059bc73fefca5e809d69b5228715ab2b70231` |
| CIFAR-10 manifest SHA-256 | `8cb47a080c42ae962fae9eeb441716691df605b781c62e18fac828e36732bd10` |
| Environment | `Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.43` |
| Python | `3.11.7` |
| Package versions | Flower 1.36.0, PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, NumPy 2.4.6, Ray 2.55.1 |

## Rules stored before training

- Use seed 42, 15 rounds and four clients. Capture the original training deltas from all four clients at rounds 1, 8 and 15 of the lossless LLZ run.
- Reproduce each reply's QSGD draw from the run seed, round, partition identifier and the fixed purpose tag. Test LLZ tolerances 0, 1 and 2 using the same draw; retain the added-error ratios for each tolerance.
- Form the candidate budgets from the 25th, 50th and 75th percentiles of the 12 `p=1` added-error ratios, using linear interpolation at full precision. Evaluate duplicate budget values once. This grid uses the same tolerance as its reference for all three candidates. Positive `p=2` remains available for each online reply when feasible.
- Select checkpoints by minimum validation loss. Require at most a 1.0 percentage-point drop from matched QSGD. Among the qualifying adaptive settings, choose the one with the fewest cumulative training upload bytes; break exact ties by the smaller budget.
- Use a fresh QSGD reference in this batch if replay does not match the historical reference. Require exact QSGD versus lossless replay before the adaptive sweep.

## Development records and confirmation plan

The completed batch links the 12 update snapshots to their measured ratios in `calibration.json`. Before adaptive training, `frozen-budgets.json` recorded the full-precision budgets and calibration checksum. The final `results-for-review.zip` contains both files, the protocol and the comparison. All three candidates missed the validation margin on seed 42, so the batch selected no policy for confirmation.

The confirmation plan called for matched QSGD and adaptive runs on seeds 43 and 44 only if the development winner also saved training upload bytes relative to lossless LLZ. The planned confirmation seeds and this additional advancement criterion are described here for traceability; they are separate from the selection rule stored in the original development protocol. Each seed's accuracy comparison uses its own matched QSGD run with the same one-point margin. The budget stays fixed and the held-out test split stays separate from selection.

The chapter's existing fixed-method CIFAR-10 results and timings come from local WSL runs. The prior HPC batch is exploratory for this comparison. Matching seed and codec settings do not ensure identical client deltas across different training environments, so the WSL sweep uses budgets calculated from its own lossless trajectory.
