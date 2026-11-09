# Current analysis for the 10 November meeting

Updated 5 November. Supersedes the preliminary Windows-pilot meeting summary with completed WSL evidence.

## Development selection

All seed-42 runs use DigiFace with 100 classes, four IID clients, 15 FedAvg rounds and one local epoch. Cached ResNet-18 layer3 features feed trainable layer4/classifier. Checkpoint selection uses minimum validation loss; test evaluation and error feedback remain disabled.

| Policy | Validation accuracy | Training upload MB | Decision |
|---|---:|---:|---|
| QSGD | 83.5% | 506.836 | Reference |
| LLZ p=0 | 83.5% | 59.814 | Exact numerical QSGD replay |
| LLZ p=1 | 79.3% | 12.481 | Exploratory fixed tolerance |
| Adaptive tau=5.9 | 83.2% | 40.924 | Pass development margin |
| Adaptive tau=6.5 | 81.1% | 25.812 | Fail development margin |
| Adaptive tau=7 | 78.9% | 14.973 | Fail development margin |

The original budgets came from quantiles of eight training-only updates, with explicit rounding. That origin is traceable but weak evidence by itself. Among the tested full-run budgets, only 5.9 passed the predeclared 1.0 pp development gate. The audit and sensitivity note is `docs/thesis/adaptive-budget-sensitivity-2026-11-03.md`.

## Frozen confirmation

| Seed | QSGD accuracy | Adaptive accuracy | Change | Adaptive uploads | Gate |
|---:|---:|---:|---:|---:|---|
| 43 | 83.6% | 83.3% | -0.3 pp | 40.165 MB | Pass |
| 44 | 83.2% | 82.0% | -1.2 pp | 39.347 MB | Fail |

Uploads fell 92.1–92.2%; total serialized objects fell about 10.2%. Each adaptive run passed the saved-diagnostic selection audit across 60 client decisions. The incomplete seed-44 attempt is excluded and its completed same-settings retry is included. Source, manifest and per-pair initialization match the frozen protocol. No budget retuning followed confirmation.

The policy does not consistently meet the accuracy criterion. This is useful negative evidence about the calibration rule, even though the codec and selection diagnostics are consistent. Validation-selected checkpoints are not held-out test performance or evidence of equivalence. Whole-launch adaptive runtimes were 46.3/56.1 min versus 11.6/12.4 min for QSGD and should be treated as descriptive.

## Literature and next decision

Adaptive compression already has precedents. Our question is the specific same-QSGD-draw, complete-reply byte minimization over LLZ tolerances under an additional-error constraint. The next proposal is a tighter or stage-dependent budget, or a total-error/alignment proxy, developed under a newly declared protocol. Matched error feedback is a later ablation worth discussing. These are proposals, not completed runs or a new convergence result.

See the current package README for exact source hashes, data files and reproduction commands. The literature review's historical pilot snapshot predates the latest experiments and the slides supply the updated status.
