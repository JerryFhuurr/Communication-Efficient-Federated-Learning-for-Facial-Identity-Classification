# Saved update walkthrough — current WSL confirmation

Use `python3 outputs/meeting-20261110-20261105/demo_saved_decision.py` from the project root. The file reads the portable saved JSON and confirmation summary, recomputes the selection from recorded diagnostics, and prints the full-run outcomes.

Source: seed 43, round 5, client 1, in `outputs/adaptive-budget-confirmation-20261103/seed-43-adaptive/fedavg-qsgd-llz-adaptive-20261103T121314626273Z/adaptive_selection.jsonl`.

1. Form local delta u. Its squared norm is 0.09296257669553568.
2. Make one QSGD draw z, shared across p candidates.
3. Locally decode the LLZ candidates. Their added squared errors are 0, 0.543928039623173 and 0.6036774843569663.
4. Divide by the original update squared norm. Ratios are 0, 5.851043 and 6.493769.
5. With tau=5.9, p=0 and p=1 are feasible. Complete replies cost 1,039,373; 208,554; and 203,378 bytes for p=0/1/2 respectively.
6. Choose p=1 because the smaller p=2 reply is infeasible. Observed serialized bytes equal 208,554.
7. The server decodes for weighted FedAvg. The full experiment's checkpoint and communication summaries are separate evidence from this one decision.

This is a saved-diagnostic walkthrough. It does not regenerate model deltas, re-encode tensors or repeat training. Earlier Windows-pilot examples remain historical. Codecs were already checked; no new training or broad validation run is needed for this meeting demonstration.
