from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "adaptive-budget-confirmation-20261103"
EXPECTED_SOURCE = (ROOT / "docs/thesis/meeting-2026-11-10/confirmation-source-sha256.txt").read_text().strip()
EXPECTED_MANIFEST = "3a138b43244bac7a0715646e981e75a03d46fa593b5a35f55234fe45868d6b52"
GATE_PP = 1.0

def load_run(seed: int, method: str) -> dict:
    parent = OUT / (f"seed-44-adaptive-retry1" if seed == 44 and method == "adaptive" else f"seed-{seed}-{method}")
    run_dirs = sorted(p for p in parent.iterdir() if p.is_dir() and (p / "experiment.json").is_file())
    if len(run_dirs) != 1:
        raise RuntimeError(f"{parent}: expected exactly one completed run directory, found {len(run_dirs)}")
    path = run_dirs[0]
    exp = json.loads((path / "experiment.json").read_text())
    if exp.get("status") != "completed":
        raise RuntimeError(f"{path}: run status is {exp.get('status')!r}")
    # Report validation accuracy at the checkpoint with lowest validation loss.
    cp = json.loads((path / "best_checkpoint.json").read_text())
    # Read the saved communication ledger for the complete run.
    comm = json.loads((path / "communication.json").read_text())["totals"]
    cfg = exp["config"]
    record = {
        "seed": seed, "method": method, "path": str(path.relative_to(ROOT)),
        "source_sha256": exp["source_sha256"], "manifest_sha256": exp["manifest_sha256"],
        "initial_model_sha256": exp["initial_model_sha256"],
        "rounds": cfg["num-server-rounds"], "test_evaluated": cfg["evaluate-final-test"],
        "budget": cfg["llz-budget"], "accuracy_percent": cp["validation_accuracy"] * 100,
        "validation_loss": cp["validation_loss"], "best_round": cp["step"],
        # Uploads sum serialized client training replies across all rounds.
        "training_upload_bytes": comm["train"]["uplink"]["serialized_object_bytes"],
        "training_upload_MB": comm["train"]["uplink"]["serialized_object_bytes"] / 1e6,
        # Total includes both directions and all recorded phases.
        # These are logical object bytes, not captured network traffic.
        "total_object_bytes": comm["total"]["serialized_object_bytes"],
        "total_object_MB": comm["total"]["serialized_object_bytes"] / 1e6,
        "elapsed_seconds": exp["elapsed_seconds"], "mean_selected_rho": None,
        "max_selected_rho": None, "p0": None, "p1": None, "p2": None,
        "selection_records": 0, "audit_errors": 0,
    }
    if record["source_sha256"] != EXPECTED_SOURCE:
        raise RuntimeError(f"{path}: source fingerprint mismatch")
    if record["manifest_sha256"] != EXPECTED_MANIFEST:
        raise RuntimeError(f"{path}: manifest fingerprint mismatch")
    if record["rounds"] != 15 or record["test_evaluated"]:
        raise RuntimeError(f"{path}: protocol mismatch: rounds/test flag")
    if method == "adaptive":
        if record["budget"] != 5.9:
            raise RuntimeError(f"{path}: unexpected budget {record['budget']}")
        rows = [json.loads(line) for line in (path / "adaptive_selection.jsonl").read_text().splitlines() if line.strip()]
        rhos, counts, errors = [], [0, 0, 0], 0
        for item in rows:
            ps, errs, sizes = item["adaptive_candidate_p"], item["adaptive_candidate_squared_error"], item["adaptive_candidate_message_bytes"]
            rho = [e / item["qsgd_input_squared_norm"] for e in errs]
            selected = int(item["adaptive_selected_p"])
            i = ps.index(float(selected))
            feasible = [j for j, value in enumerate(rho) if value <= record["budget"] + 1e-9]
            cheapest = min(feasible, key=lambda j: (sizes[j], ps[j]))
            if rho[i] > record["budget"] + 1e-9 or ps[cheapest] != selected or int(sizes[i]) != int(item["observed_message_bytes"]):
                errors += 1
            rhos.append(rho[i])
            counts[selected] += 1
        if len(rows) != 60:
            errors += 1
        record.update(mean_selected_rho=sum(rhos)/len(rhos), max_selected_rho=max(rhos),
                      p0=counts[0], p1=counts[1], p2=counts[2], selection_records=len(rows), audit_errors=errors)
        if errors:
            raise RuntimeError(f"{path}: {errors} adaptive-selection audit errors")
    elif record["budget"] != 0:
        raise RuntimeError(f"{path}: QSGD must use budget 0")
    return record

records = [load_run(seed, method) for seed in (43, 44) for method in ("qsgd", "adaptive")]
for seed in (43, 44):
    pair = [r for r in records if r["seed"] == seed]
    qsgd, adaptive = pair
    if qsgd["initial_model_sha256"] != adaptive["initial_model_sha256"]:
        raise RuntimeError(f"seed {seed}: matched initial model hashes differ")
    if qsgd["source_sha256"] != adaptive["source_sha256"]:
        raise RuntimeError(f"seed {seed}: matched source hashes differ")
    diff = adaptive["accuracy_percent"] - qsgd["accuracy_percent"]
    adaptive["accuracy_delta_vs_qsgd_pp"] = diff
    adaptive["gate"] = "PASS" if diff >= -GATE_PP else "FAIL"
    qsgd["accuracy_delta_vs_qsgd_pp"] = 0.0
    qsgd["gate"] = "reference"
fields = list(dict.fromkeys([*records[0], "accuracy_delta_vs_qsgd_pp", "gate"]))
csv_path = OUT / "confirmation-summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
summary = {
    "status": "complete", "expected_source_sha256": EXPECTED_SOURCE,
    "expected_manifest_sha256": EXPECTED_MANIFEST, "gate_drop_pp": GATE_PP,
    "records": records,
}
(OUT / "confirmation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Confirmation audit complete")
for seed in (43, 44):
    qsgd, adaptive = [r for r in records if r["seed"] == seed]
    print(f"seed {seed}: QSGD={qsgd['accuracy_percent']:.2f}% ({qsgd['training_upload_MB']:.3f} MB upload); "
          f"adaptive={adaptive['accuracy_percent']:.2f}% ({adaptive['training_upload_MB']:.3f} MB upload), "
          f"delta={adaptive['accuracy_delta_vs_qsgd_pp']:+.2f} pp, gate={adaptive['gate']}, "
          f"mean/max rho={adaptive['mean_selected_rho']:.3f}/{adaptive['max_selected_rho']:.3f}, "
          f"selection audit errors={adaptive['audit_errors']}")
print(f"Saved {csv_path}")
