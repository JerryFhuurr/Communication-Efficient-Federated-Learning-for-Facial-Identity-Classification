# Frozen confirmation protocol: adaptive LLZ tau=5.9

Frozen on 3 November 2026 before confirmation training. This protocol must not
be changed after inspecting seed-43/44 validation outcomes.

## Frozen setup

- Development-selected policy: adaptive QSGD+LLZ with tau=5.9.
- Confirmation seeds: 43 and 44.
- Matched methods at each seed: QSGD, then adaptive tau=5.9.
- Acceptance gate: adaptive validation accuracy is at most 1.0 percentage
  point below matched QSGD on each confirmation seed. Report each pair.
- Data: Digiface frozen ResNet-18 layer-3 feature manifest,
  data/digiface-resnet18-layer3/manifest.json; manifest SHA-256:
  3a138b43244bac7a0715646e981e75a03d46fa593b5a35f55234fe45868d6b52.
- Task/model: flower_face.finetune_task, 100 identity classes, 224 input
  configuration, the same seed-deterministic initialization rule, no
  separately supplied initial checkpoint.
- FL: four IID clients, FedAvg, 15 rounds, one local epoch, batch size 16.
- Optimizer: learning rate 0.003, weight decay 0, horizontal-flip augmentation.
- QSGD: 127 levels. Adaptive LLZ: p candidates 0/1/2, window 128,
  one choice per client update, minimum complete serialized reply subject to
  the frozen distortion budget 5.9.
- Validation: lowest validation loss selects the checkpoint, earliest tie.
  Test evaluation is disabled. Error feedback remains disabled.
- Environment: WSL2, Python 3.11.7, Flower 1.36.0, PyTorch 2.6.0+cpu,
  torchvision 0.21.0+cpu, NumPy 2.4.6, Ray 2.55.1; methods run sequentially.

## Frozen source identity

Git HEAD: 1b0e2cc315153f1ae2b9dfcbbf03b4de9f59b939.
Runner source fingerprint (the runner's own SHA-256 over pyproject.toml and
Python files in flower_face/, compression/, and federated_compression/):
0bfa8e5368114e3f73cf16747d5713ae859f61be0be2b472f38a9787d79c5bb5.

The working tree has other documentation and project changes, so Git HEAD alone
does not identify the executed code. The runner fingerprint is the operative
identity. The run script checks it before and after every job and stops if it
changes. The same source fingerprint must appear in all four experiment.json
files. Within each seed, QSGD and adaptive runs must also have the same
initial_model_sha256 and manifest_sha256.

The results are confirmation evidence for the policy selected on seed 42.
Do not adjust tau or other settings in response to these seeds. If a gate fails,
report the failure and reserve any revised policy for a new, separately
predeclared seed.

## Run and output

Run from the project root:

    bash experiments/run_adaptive_budget_confirmation.sh

The four jobs execute sequentially in this order: seed 43 QSGD, seed 43
adaptive, seed 44 QSGD, seed 44 adaptive. Outputs, stdout logs, and progress
records are under outputs/adaptive-budget-confirmation-20261103/.

## Confirmation results (3 November 2026)

All four valid runs completed 15 rounds. Their runner source fingerprint,
manifest fingerprint, and matched initial-model hashes were identical within
the protocol. The incomplete first seed-44 adaptive attempt is excluded; its
same-settings retry completed and is the seed-44 adaptive result below.

| Seed | Method | Best validation accuracy | Training uplink | Adaptive distortion | Gate vs matched QSGD |
|---:|---|---:|---:|---:|---|
| 43 | QSGD | 83.6% | 506.836 MB | — | Reference |
| 43 | Adaptive, tau=5.9 | 83.3% | 40.165 MB | mean rho 2.400; max 5.894 | Pass, -0.3 pp |
| 44 | QSGD | 83.2% | 506.836 MB | — | Reference |
| 44 | Adaptive, tau=5.9 | 82.0% | 39.347 MB | mean rho 2.520; max 5.857 | Fail, -1.2 pp |

Adaptive uplink was reduced by 92.1% and 92.2% versus matched QSGD for seeds
43 and 44, respectively. The per-seed 1.0 pp accuracy gate passed seed 43 and
failed seed 44. Thus tau=5.9 shows a large communication reduction in these
runs, but does not pass the predeclared accuracy criterion consistently; do
not claim robust accuracy preservation or tune tau using these confirmation
seeds. These are best-validation-checkpoint accuracies, not held-out test
accuracies. The selection audit found 0 errors across 60 client decisions for
each adaptive run, with every selected distortion within budget.

The machine-readable audit is in
outputs/adaptive-budget-confirmation-20261103/confirmation-summary.json and
.csv. Per-run logs and metrics remain in that output directory.

## Administrative recovery record

The first seed-44 adaptive attempt ended with status `incomplete`: it recorded
validation through round 13 and began round 14 without producing a completed
manifest. The launcher had treated a zero exit code as success; its completion
check was tightened. The exact frozen configuration was rerun to a separate
`seed-44-adaptive-retry1/` directory. Only the completed retry is included in
the confirmation summary; the earlier attempt is retained for provenance.
