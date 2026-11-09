from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "adaptive-budget-sweep-20261103"
RUNS = {
    "QSGD": ("adaptive-dev-seed42-wsl-20261102/qsgd-attempt-1/fedavg-qsgd-20261102T100306980744Z", None),
    "Adaptive tau=5.9": ("adaptive-dev-seed42-wsl-20261102/adaptive-tight-attempt-1/fedavg-qsgd-llz-adaptive-20261102T133649918726Z", 5.9),
    "Adaptive tau=6.5": ("adaptive-budget-sweep-20261103/tau-6-5/fedavg-qsgd-llz-adaptive-20261103T103241929884Z", 6.5),
    "Adaptive tau=7": ("adaptive-budget-sweep-20261103/tau-7/fedavg-qsgd-llz-adaptive-20261103T094626706389Z", 7.0),
}
rows = []
for label, (relative, budget) in RUNS.items():
    run = ROOT / "outputs" / relative
    checkpoint = json.loads((run / "best_checkpoint.json").read_text())
    communication = json.loads((run / "communication.json").read_text())["totals"]
    uploads = communication["train"]["uplink"]["serialized_object_bytes"]
    record = {
        "method": label,
        "budget": budget,
        "validation_accuracy_percent": checkpoint["validation_accuracy"] * 100,
        "validation_loss": checkpoint["validation_loss"],
        "training_upload_serialized_bytes": uploads,
        "training_upload_MB": uploads / 1e6,
        "total_serialized_bytes": communication["total"]["serialized_object_bytes"],
        "mean_selected_rho": None,
        "max_selected_rho": None,
        "selected_p0": None,
        "selected_p1": None,
        "selected_p2": None,
        "selection_seconds_sum": None,
        "records": 0,
    }
    selections_path = run / "adaptive_selection.jsonl"
    if selections_path.exists():
        selected_rhos, counts, seconds = [], [0, 0, 0], 0.0
        for line in selections_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            ps = item["adaptive_candidate_p"]
            errs = item["adaptive_candidate_squared_error"]
            sizes = item["adaptive_candidate_message_bytes"]
            rhos = [e / item["qsgd_input_squared_norm"] for e in errs]
            selected_p = int(item["adaptive_selected_p"])
            index = ps.index(float(selected_p))
            assert rhos[index] <= item["adaptive_budget"] + 1e-9
            feasible = [i for i, rho in enumerate(rhos) if rho <= item["adaptive_budget"] + 1e-9]
            cheapest = min(feasible, key=lambda i: (sizes[i], ps[i]))
            assert ps[cheapest] == selected_p
            assert int(sizes[index]) == int(item["observed_message_bytes"])
            selected_rhos.append(rhos[index])
            counts[selected_p] += 1
            seconds += item["adaptive_selection_seconds"]
        record.update({
            "mean_selected_rho": sum(selected_rhos) / len(selected_rhos),
            "max_selected_rho": max(selected_rhos),
            "selected_p0": counts[0], "selected_p1": counts[1], "selected_p2": counts[2],
            "selection_seconds_sum": seconds, "records": len(selected_rhos),
        })
    rows.append(record)

OUT.mkdir(parents=True, exist_ok=True)
csv_path = OUT / "full-run-summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

fig, ax = plt.subplots(figsize=(10.4, 6.4), constrained_layout=True)
ax.axhline(82.5, color="#b23a48", linestyle="--", linewidth=1.5, label="1.0 pp development gate")
qsgd = rows[0]
ax.scatter(qsgd["training_upload_MB"], qsgd["validation_accuracy_percent"], marker="*", s=190,
           color="#525866", edgecolor="white", linewidth=0.7, zorder=4, label="QSGD reference")
adaptive = rows[1:]
scatter = ax.scatter([r["training_upload_MB"] for r in adaptive],
                     [r["validation_accuracy_percent"] for r in adaptive],
                     c=[r["mean_selected_rho"] for r in adaptive], cmap="viridis",
                     s=110, edgecolor="black", linewidth=0.7, zorder=5,
                     vmin=0, vmax=7, label="Adaptive LLZ")
for r, offset in zip(rows, [(8, 8), (8, -21), (8, 8), (8, 8)]):
    ax.annotate(f'{r["method"]}\n{r["validation_accuracy_percent"]:.1f}%, {r["training_upload_MB"]:.1f} MB',
                (r["training_upload_MB"], r["validation_accuracy_percent"]),
                xytext=offset, textcoords="offset points", fontsize=8.5)
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Mean selected relative squared distortion, rho")
ax.set_xscale("log")
ax.set_xlabel("Training upload (serialized Flower objects, MB; log scale)")
ax.set_ylabel("Best validation accuracy (%)")
ax.set_title("Adaptive LLZ budget sweep: seed 42, 15 rounds")
ax.grid(True, which="both", alpha=0.22)
ax.set_ylim(77.5, 85.2)
ax.legend(loc="lower right", frameon=True)
fig.text(0.01, -0.02, "One seed; validation checkpoint selected by minimum loss. Development comparison, not a generalization guarantee.",
         fontsize=8, color="#444")
png_path = OUT / "full-run-tradeoff.png"
fig.savefig(png_path, dpi=180, bbox_inches="tight")
print("Saved", png_path)
print("Saved", csv_path)
for row in rows:
    print(row)
