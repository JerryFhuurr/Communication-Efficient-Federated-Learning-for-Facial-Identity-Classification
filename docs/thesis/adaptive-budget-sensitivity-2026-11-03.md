# Adaptive LLZ distortion-budget audit and sensitivity

Updated 3 November 2026. This note separates the reproducible origin of the
historical values `tau=5.9` and `tau=7.0` from evidence about their learning
quality. It records the values' origin, a frozen-update replay, and matched WSL seed-42
runs at tau=5.9, 6.5 and 7. No held-out test evaluation is used.

## What the budget measures

For each client update `u`, the implementation draws QSGD once and decodes it
to `z`. Each LLZ tolerance candidate `p` is encoded and decoded locally to
`z_p`. The selector computes

```
rho(p) = sum_t ||z_p,t - z_t||_2^2 / sum_t ||u_t||_2^2
```

It keeps candidates with `rho(p) <= tau`, then selects the lowest complete
serialized reply cost; exact byte ties prefer lower `p`. `p=0` is a lossless
fallback relative to decoded QSGD. Thus `tau` is a dimensionless *relative
squared-error* limit for the second LLZ stage. It is not LLZ's `p`, a percent,
a limit on total QSGD-plus-LLZ error, or a validation-accuracy tolerance. The
codec tolerance `p` and adaptive error budget `tau` are distinct: the IEEE
paper's empirical p=1 result does not imply a suitable `tau` for this model. In
our one-seed WSL comparison, fixed p=1 reached 79.3% validation accuracy vs
83.5% for QSGD, whereas adaptive `tau=5.9` reached 83.2%; these are results
for this demo, not a refutation of the paper's findings.

Under this normalization, `tau=5.9` permits the added LLZ reconstruction-error
norm to be as large as `sqrt(5.9) = 2.43` times the full-precision update norm;
`tau=7` permits `sqrt(7) = 2.65` times that norm. These large ratios are why the
values need empirical justification and should not be described as inherently
safe.

## Where 5.9 and 7 came from

The frozen calibration record is
`outputs/adaptive-study-20261021T104244874038Z/study.json`; its source is the
training-only update collection at
`outputs/ae-calibration-20261005T122126176771Z/collection.json`. For each of
four clients, the calibration took two snapshots (steps 24 and 47) from an
independently reset one-epoch local trajectory: eight updates total. The
collection protocol states that its image source is the training split only;
“validation” is an update-reconstruction role name, not identity-validation
data. These were not federated rounds, and they were not eight independent
clients or seeds.

| Client / local step | Candidate `p=1` ratio | Candidate `p=2` ratio |
|---|---:|---:|
| 0 / 24 | 5.8003 | 6.3168 |
| 0 / 47 | 6.4212 | 7.0581 |
| 1 / 24 | 6.0997 | 6.5680 |
| 1 / 47 | 6.0747 | 6.5651 |
| 2 / 24 | 5.9484 | 6.7914 |
| 2 / 47 | 6.8097 | 7.3720 |
| 3 / 24 | 5.8493 | 6.4434 |
| 3 / 47 | 6.2342 | 6.8854 |

The calibration code applied a stated quantile heuristic:

- `tau_tight = floor_2sigfig(Q25(p=1)) = floor_2sigfig(5.923659) = 5.9`.
- `tau_loose = ceil_2sigfig(max(Q75(p=2), 1.1*tau_tight)) = ceil_2sigfig(max(6.928620, 6.49)) = 7.0`.

This gives a traceable *pilot calibration rule*: roughly the lower quartile of
p=1 ratios and upper quartile of p=2 ratios. It is not a paper-prescribed
constant, an accuracy-derived boundary, or a convergence guarantee. With only
eight clustered training updates, the quantiles are exploratory and have no
useful inferential precision.

## How each full-run result is produced

1. Restore the fixed seed-42 initial model and the same 100-class training/validation manifest; each of four clients receives its fixed IID partition.
2. For every communication round, each client trains for one local epoch and forms its model delta `u`.
3. Draw QSGD once, then locally build/decode the three LLZ candidates `p=0,1,2`. Measure each candidate's added squared error and complete serialized reply bytes.
4. Keep candidates meeting `rho <= tau`; send the feasible candidate with the smallest reply. The server checks the selection diagnostics and byte count, decodes, and performs sample-weighted FedAvg.
5. Repeat for 15 rounds. Evaluate validation loss/accuracy after each round; report the checkpoint with the lowest validation loss (earliest tie). Held-out test evaluation is disabled.
6. Sum server-ledger serialized object bytes separately for training uploads and all directions/phases. These are logical Flower object sizes, not packet-captured network traffic.

The saved p=0 replay also verifies that this second stage reconstructs the QSGD values exactly, so the p=0 learning trajectory matches QSGD. The budget sweep replay later in this note reuses saved candidate arrays and byte costs; it does not repeat steps 1–5 for each budget.

## Evidence from the current seed-42 run

The WSL development protocol fixes the DigiFace feature dataset, 100 identities,
four IID clients, 15 FedAvg rounds, one local epoch, QSGD levels 127, LLZ window
128, seed 42, validation checkpoint selection, and no test-set evaluation.
The `tau=5.9` adaptive run produced 83.2% best validation accuracy, versus
83.5% for matched QSGD: a 0.3 percentage-point drop, inside the predeclared
single-seed 1.0-point development gate. Adaptive training uploads were 40.924
MB versus 506.836 MB for QSGD (91.93% lower); total serialized-object bytes
were 4098.730 MB versus 4564.642 MB (10.2% lower). This supports one development
operating point, not an accuracy guarantee or statistical-equivalence claim.

The maximum selected ratio `5.8947` is evidence that the implementation
respected the configured constraint on all 60 client-round updates. It does
not independently justify 5.9: selecting only candidates under the limit
makes every selected maximum at most `tau` by construction. Its 0.0053 margin
also shows one selected update sat very near the boundary.

## Measured full-run trade-off

All points below are actual 15-round WSL seed-42 runs with the same 100-class
DigiFace feature manifest, four IID clients, initial model, QSGD level 127,
LLZ window 128, optimizer settings, and best-checkpoint rule. The predeclared
single-seed development gate is at least 82.5% validation accuracy (no more
than 1.0 percentage point below QSGD's 83.5%). The table reports complete
serialized training uploads, selected distortion, and the candidate choices.

| Policy | Best validation accuracy | Training upload (MB) | Mean / max selected rho | p=0 / 1 / 2 choices | Gate |
|---|---:|---:|---:|---:|---|
| QSGD | 83.5% | 506.836 | — | — | reference |
| Adaptive tau=5.9 | 83.2% | 40.924 | 2.337 / 5.895 | 35 / 18 / 7 | pass by 0.7 pp |
| Adaptive tau=6.5 | 81.1% | 25.812 | 4.514 / 6.485 | 16 / 32 / 12 | fail by 1.4 pp |
| Adaptive tau=7 | 78.9% | 14.973 | 6.286 / 6.973 | 3 / 41 / 16 | fail by 3.6 pp |

At tau=5.9, training uploads are 91.93% lower than QSGD while validation
accuracy falls by 0.3 point. Raising the budget to 6.5 saves another 36.9% of
the upload bytes versus 5.9, but loses 2.1 more accuracy points and fails the
gate. Raising it to 7 saves 63.4% versus 5.9, but loses 4.3 more points. The
complete serialized-object totals are 4564.642 MB for QSGD, 4098.730 MB for
5.9, 4083.618 MB for 6.5 and 4072.778 MB for 7: downlink model objects dominate
these totals, so the much larger training-upload reductions translate into
about 10.2–10.8% total-object savings. These are Flower logical object sizes,
not packet-captured network traffic.

The 5.9 policy is the only tested adaptive budget that meets the frozen
single-seed quality gate. Its calibration origin (the lower-quartile p=1 ratio
from eight training-update snapshots) makes it a traceable starting point; the
matched run makes it the strongest current development choice among the tested
values. The 6.5 and 7 results show that allowing more distortion bought more
upload reduction at a material validation-accuracy cost. The data do not prove
5.9 is universally optimal or safe: all three adaptive runs use one seed, and
the original eight calibration records are clustered snapshots. Keep 5.9
provisional until locked settings are confirmed on additional seeds.

![Full-run budget trade-off: validation accuracy, upload size, and mean selected distortion](../../outputs/adaptive-budget-sweep-20261103/full-run-tradeoff.png)

The plot uses a log scale for upload size and color for the mean selected
relative squared distortion; its dashed horizontal line is the 82.5% gate.
Machine-readable values are in
[full-run-summary.csv](../../outputs/adaptive-budget-sweep-20261103/full-run-summary.csv).
The plot is reproducibly generated by
[experiments/plot_adaptive_budget_sweep.py](../../experiments/plot_adaptive_budget_sweep.py).

## Can theory supply a safer replacement?

The project’s conditional-stationarity note derives an idealized sufficient
condition for a full-gradient, one-step model with `tau < 1` and additional
alignment/moment assumptions. It does not establish those assumptions for this
shuffled, one-local-epoch neural FedAvg run, so it cannot certify any operational
budget here. In the frozen-update replay, even `tau=0.5` selects p=0 for all 60
updates. A theory-inspired sub-unit budget would therefore remove the measured
adaptive p>0 benefit on this trace; it is a useful conservative reference, not
an evidence-based replacement threshold for this experiment.

## Frozen-update sensitivity replay

To cheaply inspect the selector response, each budget below was replayed over
the 60 saved candidate sets from the completed `tau=5.9` WSL run. The replay
reselects among the already encoded `p=0/1/2` candidate costs; it does not
retrain FedAvg. Its byte curve therefore estimates local selector sensitivity
on this realized trajectory, not the full-training outcome at another `tau`.

| `tau` | Replayed p counts (0 / 1 / 2) | Replayed uplink reply bytes | Mean selected `rho` | Maximum selected `rho` |
|---:|---:|---:|---:|---:|
| 0–4 | 60 / 0 / 0 | 60.463 MB | 0.000 | 0.000 |
| 5 | 59 / 0 / 1 | 59.759 MB | 0.078 | 4.685 |
| 5.9 | 35 / 18 / 7 | 40.924 MB | 2.337 | 5.895 |
| 6 | 32 / 19 / 9 | 38.482 MB | 2.653 | 5.966 |
| 6.5 | 6 / 26 / 28 | 17.413 MB | 5.510 | 6.417 |
| 7 | 0 / 9 / 51 | 12.252 MB | 6.386 | 6.985 |
| 8–10 | 0 / 0 / 60 | 12.204 MB | 6.488 | 7.426 |

The sharp step between `tau=5` and `5.9` reflects the observed candidate-error
ratios, not a smooth universal law. The `tau=7` replay nearly saturates its
limit on a different set of updates and would select p=2 for 51 of 60 records.
That higher compression is paired with more added distortion. Accuracy cannot
be inferred from this offline replay.

![Offline selector replay: communication and distortion on frozen updates](../../outputs/adaptive-budget-sweep-20261103/trace-replay/trace-replay.png)

## Full-run comparison and decision rule

The matched 15-round WSL tau=6.5 run's best checkpoint was round 13 and
reached 81.1% validation accuracy; it failed the gate by 1.4 points. It sent
25.812 MB of training uploads and 4083.618 MB of total serialized Flower
objects. Its selected p=0/1/2 counts were 16/32/12, with mean/max selected
ratios 4.514/6.485. The matched tau=7 run's best checkpoint was round 13 and
reached 78.9%, a 4.6-point drop from QSGD and a 3.6-point gate failure. It sent
14.973 MB of training uploads and 4072.778 MB total; its p counts were 3/41/16
and mean/max ratios 6.286/6.973.

For each of the three adaptive runs, an audit recomputed the selected
distortion from the saved candidate errors and update norm, confirmed every
selection was feasible, recomputed the least-byte feasible candidate, and
matched selected bytes to the saved Flower message record. Each audit covered
all 60 client-round choices (240 communication ledger messages across train
and validation). There were zero budget, selection, or selected-size
mismatches. This establishes that the implementation applied the declared
rule consistently; it does not establish that the rule preserves accuracy in
general.

The decision rule was: choose the largest tested budget meeting the
predeclared validation gate, then lock it and confirm on seeds 43 and 44. In
this small sweep, 5.9 is the selected provisional operating point; 6.5 and 7
are useful rejected comparisons that quantify the additional compression and
accuracy cost. Do not describe 5.9 as proved optimal or as a convergence-safe
constant. Before a thesis-level claim, rerun QSGD and the locked adaptive
setting on the confirmation seeds and report per-seed outcomes without tuning
the budget on those confirmation results.

## Reproduction

The full-run comparison plot and CSV are rebuilt from the project root with:

```bash
.venv/bin/python experiments/plot_adaptive_budget_sweep.py
```

To reproduce an adaptive run, replace TAU with 5.9, 6.5 or 7, and use a fresh
output directory:

```bash
.venv/bin/python -m flower_face.run --rounds 15 --manifest data/digiface-resnet18-layer3/manifest.json --task-module flower_face.finetune_task --num-classes 100 --num-clients 4 --image-size 224 --seed 42 --lr 0.003 --weight-decay 0 --augmentation horizontal-flip --compression qsgd-llz-adaptive --qsgd-levels 127 --llz-p 0 --llz-window 128 --llz-budget TAU --skip-test --output-dir outputs/adaptive-budget-sweep-20261103/reproduction
```

The calibration rule, update-level candidates, configurations, and round
metrics remain in the study JSON files and each run's experiment.json,
history.jsonl, adaptive_selection.jsonl, and communication ledger.
