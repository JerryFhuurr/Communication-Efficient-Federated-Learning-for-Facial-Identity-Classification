# Reproducing the 10 November meeting evidence

Updated 5 November. See `outputs/meeting-20261110-20261105/README.md` for verified source identity, exact saved-run locations and commands.

# Supervisor meeting — 10 November 2026

Prepared 5 November. Use the first seven slides for the five-minute talk; slides 8–9 support questions. Speaking notes are embedded in the PowerPoint and provided separately with Q&A.

## Files

- `adaptive-llz-meeting-20261110.pptx`: editable presentation with literature table, native budget chart, saved-update example and confirmation results.
- `presentation-notes.md`: timed script, likely questions, questions for the supervisor.
- `demo_saved_decision.py` and `saved-decision.json`: portable saved-record walkthrough.
- `confirmation-summary.json` and `budget-sweep.csv`: machine-readable data used in the slides.
- `frozen-experiment-source.zip`: Python packages, TOML configuration and relevant protocol/analysis documents. No datasets, virtual environment, credentials or model checkpoints.
- `source-sha256.txt`: operative confirmation code fingerprint, verified against the archived Python/TOML snapshot.

## Main result

Development seed 42 selected tau=5.9 among 5.9/6.5/7 using a maximum 1.0 percentage-point validation-accuracy drop. Freeze came before seeds 43/44. Seed 43 passed (-0.3 pp); seed 44 failed (-1.2 pp). Both reduced serialized training uploads by about 92%. Do not describe the policy as consistently preserving accuracy under the predeclared criterion. The completed seed-44 retry replaces an incomplete attempt without changing settings.

## Five-minute order

1. Research question and result (20 seconds)
2. Literature comparison (45 seconds)
3. Saved selection example (45 seconds)
4. Origin of 5.9 and 7 (40 seconds)
5. Seed-42 sensitivity study (50 seconds)
6. Frozen confirmation (60 seconds)
7. Interpretation and next decision (40 seconds)

## Demonstration commands in WSL

```bash
cd /home/jerrycao/projects/demo-new
python3 outputs/meeting-20261110-20261105/demo_saved_decision.py
cat outputs/adaptive-budget-confirmation-20261103/confirmation-summary.csv
```

The demo uses only the standard library and saved diagnostics. It does not train or regenerate candidate tensors. For a step-by-step explanation: identify local delta u, share one QSGD draw z across candidates, compute rho from each saved extra error, compare rho to tau, select the smallest feasible complete message, match observed bytes, and open the full-run summary. Round 5/client 1 selects p=1 because p=2 exceeds tau despite being smaller.

## Rebuild analysis from existing runs

```bash
.venv/bin/python experiments/plot_adaptive_budget_sweep.py
.venv/bin/python experiments/summarize_adaptive_budget_confirmation.py
```

These commands reread stored runs. Full training reproduction is optional and can take hours:

```bash
bash experiments/run_adaptive_budget_confirmation.sh
```

Read the launch script and use a fresh destination before deliberately rerunning training. The existing evidence is sufficient for the meeting.

## Evidence locations

- Development controls: `outputs/adaptive-dev-seed42-wsl-20261102/summary.json` and `study.json`.
- Budget comparison: `outputs/adaptive-budget-sweep-20261103/full-run-summary.csv`.
- Confirmation: `outputs/adaptive-budget-confirmation-20261103/confirmation-summary.json`.
- Original demonstration row: `outputs/adaptive-budget-confirmation-20261103/seed-43-adaptive/fedavg-qsgd-llz-adaptive-20261103T121314626273Z/adaptive_selection.jsonl`, round 5/client 1.
- Each run retains `experiment.json`, `best_checkpoint.json`, validation history, communication ledger and adaptive selections.
- Frozen protocol: `docs/thesis/meeting-2026-11-10/confirmation-protocol.md`.
- Budget origin and interpretation: `docs/thesis/adaptive-budget-sensitivity-2026-11-03.md`.
- Literature review: https://www.overleaf.com/project/6aeb378d04c7d5dba6f6b8f9

The review's research cutoff and pilot snapshot precede the November 3 sweep and confirmation. The meeting slides supplement that snapshot. The review's statements that confirmation is pending are historical status and should be updated before presenting it as a current experiment report. Export the latest reviewed Overleaf PDF for sharing.

## Frozen experiment settings

DigiFace, 100 identities; cached ResNet-18 layer3 features with trainable layer4/classifier; four IID clients; FedAvg; 15 rounds; one local epoch; batch size 16; learning rate 0.003; weight decay 0; horizontal flips. QSGD levels 127, LLZ window 128 and p candidates 0/1/2. Whole-update added LLZ squared error divided by original update squared norm. No error feedback. Lowest validation loss selects checkpoint, earliest tie. Held-out test evaluation disabled. WSL2 CPU runs execute sequentially under the 8 GB RAM setting.

Source SHA-256: `0bfa8e5368114e3f73cf16747d5713ae859f61be0be2b472f38a9787d79c5bb5`.
Manifest SHA-256: `3a138b43244bac7a0715646e981e75a03d46fa593b5a35f55234fe45868d6b52`.
Git HEAD: `1b0e2cc315153f1ae2b9dfcbbf03b4de9f59b939`; working-tree changes mean HEAD alone does not identify the executed source.

The source archive contains `SOURCE_IDENTITY.json` with per-file hashes. Full reproduction also requires local data/features and the recorded environment. Bytes are serialized Flower objects including metadata, not captured transport traffic. Times are whole-launch measurements, not isolated randomized codec benchmarks.

## Before sharing on 9 November evening

- Review the presentation and rehearse the first seven slides with the five-minute script.
- Run the short saved-record demonstration once.
- Export the latest literature PDF from Overleaf and confirm whether it has already been shared.
- Send the slides and source archive to the supervisor. This package sends nothing externally.
