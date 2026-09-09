"""Run a resumable paired study of FedAvg, QSGD, and QSGD+LLZ tolerances."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import statistics

from federated_compression import compression_settings
from flower_face.check_llz import compare as compare_lossless_pair
from flower_face.reproducibility import source_hash
from flower_face.run import default_config, launch
from flower_face.study import check_pair, collect, snapshot, write_json


BASE_METHODS = ("none", "qsgd")


def llz_label(p):
    return f"qsgd-llz-p{p}"


def run_label(run):
    method = run["config"]["compression"]
    return llz_label(run["config"]["llz-p"]) if method == "qsgd-llz" else method


def method_labels(llz_p_values):
    return (*BASE_METHODS, *(llz_label(p) for p in llz_p_values))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stats(values):
    if not values or any(value is None for value in values):
        return None
    return dict(mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values) > 1 else None)


def _check_lossy_pair(reference, candidate, p):
    """Require identical training conditions except compression method and LLZ p."""
    for key in ("manifest_sha256", "source_sha256", "versions", "platform",
                "reproducibility", "initial_model_sha256", "selection_rule",
                "communication_measurement"):
        if reference["metadata"][key] != candidate["metadata"][key]:
            raise RuntimeError(f"Lossy run pair mismatch: {key}")
    excluded = {"output-dir", "compression", "llz-p"}
    left = {key: value for key, value in reference["config"].items() if key not in excluded}
    right = {key: value for key, value in candidate["config"].items() if key not in excluded}
    if left != right or candidate["config"]["compression"] != "qsgd-llz" or candidate["config"]["llz-p"] != p:
        raise RuntimeError("Lossy comparison contains a training or codec confound")


def _mean_round_metric(run, key):
    values = [row["train"][key] for row in run["replay"] if key in row["train"]]
    return statistics.mean(values) if values else None


def summarize(runs, seeds, llz_p_values=(0,)):
    """Validate the complete paired matrix and calculate thesis-facing metrics."""
    labels = method_labels(llz_p_values)
    observed = [(r["config"]["seed"], run_label(r)) for r in runs]
    expected = {(seed, method) for seed in seeds for method in labels}
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("Expected exactly one completed run per seed, method, and LLZ tolerance")

    paired = []
    lossless = []
    for seed in seeds:
        methods = {run_label(r): r for r in runs
                   if r["config"]["seed"] == seed}
        base, qsgd = (methods[name] for name in BASE_METHODS)
        check_pair(base, qsgd)
        row = dict(
            seed=seed,
            outputs={name: methods[name]["output"] for name in labels},
            selected={name: methods[name]["best"] for name in labels},
            total_serialized_bytes={name: methods[name]["total_bytes"] for name in labels},
            training_upload_bytes={name: methods[name]["train_upload_bytes"] for name in labels},
            qsgd_upload_reduction_vs_none_percent=
                100 * (1 - qsgd["train_upload_bytes"] / base["train_upload_bytes"]),
            qsgd_total_reduction_vs_none_percent=
                100 * (1 - qsgd["total_bytes"] / base["total_bytes"]),
            qsgd_selected_accuracy_difference_pp=
                100 * (qsgd["best"]["validation_accuracy"] - base["best"]["validation_accuracy"]),
            qsgd_selected_loss_difference=
                qsgd["best"]["validation_loss"] - base["best"]["validation_loss"],
            llz={})
        for p in llz_p_values:
            label, llz = llz_label(p), methods[llz_label(p)]
            _check_lossy_pair(qsgd, llz, p)
            exact = compare_lossless_pair(qsgd, llz) if p == 0 else None
            if exact is not None:
                lossless.append(dict(seed=seed, **exact))
            row["llz"][label] = dict(
                p=p,
                upload_reduction_vs_none_percent=
                    100 * (1 - llz["train_upload_bytes"] / base["train_upload_bytes"]),
                upload_reduction_vs_qsgd_percent=
                    100 * (1 - llz["train_upload_bytes"] / qsgd["train_upload_bytes"]),
                total_reduction_vs_none_percent=
                    100 * (1 - llz["total_bytes"] / base["total_bytes"]),
                total_reduction_vs_qsgd_percent=
                    100 * (1 - llz["total_bytes"] / qsgd["total_bytes"]),
                selected_accuracy_difference_vs_qsgd_pp=100 * (
                    llz["best"]["validation_accuracy"] - qsgd["best"]["validation_accuracy"]),
                selected_loss_difference_vs_qsgd=(
                    llz["best"]["validation_loss"] - qsgd["best"]["validation_loss"]),
                mean_round_qsgd_relative_squared_error=
                    _mean_round_metric(llz, "qsgd_relative_squared_error"),
                mean_round_llz_relative_squared_error=
                    _mean_round_metric(llz, "llz_relative_squared_error"),
                exact_qsgd_replay=(exact is not None),
            )
        paired.append(row)

    methods = {}
    for method in labels:
        selected = [r for r in runs if run_label(r) == method]
        methods[method] = dict(
            n_seeds=len(selected),
            selected_validation_accuracy=_stats(
                [r["best"]["validation_accuracy"] for r in selected]),
            selected_validation_loss=_stats(
                [r["best"]["validation_loss"] for r in selected]),
            final_validation_accuracy=_stats(
                [r["final_validation"]["eval_acc"] for r in selected]),
            training_upload_bytes=_stats([r["train_upload_bytes"] for r in selected]),
            total_serialized_bytes=_stats([r["total_bytes"] for r in selected]),
            mean_round_qsgd_relative_squared_error=_stats([
                _mean_round_metric(r, "qsgd_relative_squared_error") for r in selected
                if _mean_round_metric(r, "qsgd_relative_squared_error") is not None]),
            mean_round_llz_relative_squared_error=_stats([
                _mean_round_metric(r, "llz_relative_squared_error") for r in selected
                if _mean_round_metric(r, "llz_relative_squared_error") is not None]),
        )

    reduction_keys = (
        "qsgd_upload_reduction_vs_none_percent",
        "qsgd_total_reduction_vs_none_percent",
        "qsgd_selected_accuracy_difference_pp",
        "qsgd_selected_loss_difference",
    )
    llz_statistics = {}
    for p in llz_p_values:
        label = llz_label(p)
        keys = tuple(paired[0]["llz"][label])
        llz_statistics[label] = {
            key: (_stats([row["llz"][label][key] for row in paired])
                  if key not in ("p", "exact_qsgd_replay") else paired[0]["llz"][label][key])
            for key in keys
        }
    return dict(
        methods=methods,
        paired_seeds=paired,
        paired_statistics={key: _stats([row[key] for row in paired]) for key in reduction_keys},
        llz_tolerance_statistics=llz_statistics,
        llz_p_values=list(llz_p_values),
        lossless_qsgd_llz=lossless,
        exact_qsgd_llz_p0_replay=(0 in llz_p_values),
        measurement=("Logical per-recipient serialized Flower objects, including packet, record, "
                     "and message metadata; not captured network traffic."),
        selection_rule="Lowest validation loss within each run; earliest round wins exact ties.",
        test_evaluated=False,
    )


def _verify_completed_runs(record, protocol):
    refreshed = []
    for saved in record["runs"]:
        current = collect(Path(saved["output"]).parent, saved["config"])
        if current != saved:
            raise RuntimeError("A completed run artifact changed after it was recorded")
        if (current["metadata"]["source_sha256"] != protocol["source_sha256"]
                or current["metadata"]["manifest_sha256"] != protocol["manifest_sha256"]):
            raise RuntimeError("Completed run source or manifest differs from the frozen study")
        refreshed.append(current)
    return refreshed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--qsgd-levels", type=int, default=127)
    parser.add_argument("--llz-p-values", type=int, nargs="+", default=[0, 1],
                        help="Distinct LLZ tolerances to compare; include 0 for the exact replay gate")
    parser.add_argument("--llz-window", type=int, default=128)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path,
                        help="Resume an interrupted study directory using its frozen app")
    args = parser.parse_args()
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= 2**32 for seed in args.seeds)):
        parser.error("seeds must be distinct integers in [0, 2**32-1]")
    if args.rounds < 1:
        parser.error("rounds must be positive")
    if (not args.llz_p_values or len(set(args.llz_p_values)) != len(args.llz_p_values)
            or any(type(p) is not int or p < 0 or p > 2 * args.qsgd_levels
                   for p in args.llz_p_values)):
        parser.error("LLZ p values must be distinct integers from 0 through twice qsgd-levels")

    root = Path(__file__).resolve().parents[1]
    if args.resume:
        output = args.resume.resolve()
        import json
        record = json.loads((output / "study.json").read_text(encoding="utf-8"))
        protocol = record["protocol"]
        if (args.seeds != protocol["seeds"] or args.rounds != protocol["rounds"]
                or args.qsgd_levels != protocol["qsgd_levels"]
                or args.llz_p_values != protocol["llz_p_values"]
                or args.llz_window != protocol["llz_window"]):
            parser.error("resume arguments differ from the saved protocol")
    else:
        parent = (args.output_dir or root / "outputs").resolve()
        output = parent / ("compression-study-" +
                           datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        output.mkdir(parents=True, exist_ok=False)
        stage = output / "app"
        snapshot(root, stage)
        config = default_config(root)
        config.update({"num-server-rounds": args.rounds, "qsgd-levels": args.qsgd_levels,
                       "llz-p": 0, "llz-window": args.llz_window,
                       "evaluate-final-test": False})
        for p in args.llz_p_values:
            compression_settings(dict(config, compression="qsgd-llz", **{"llz-p": p}))
        protocol = dict(
            seeds=args.seeds, rounds=args.rounds,
            methods=list(method_labels(args.llz_p_values)),
            qsgd_levels=args.qsgd_levels, llz_p_values=args.llz_p_values,
            llz_window=args.llz_window,
            base_config=config, source_sha256=source_hash(stage),
            manifest_sha256=file_hash(config["manifest"]),
            framework=("Client model deltas; QSGD uses one L2 norm per tensor; each LLZ "
                       "tolerance encodes the same deterministic QSGD draw; downlink is uncompressed."),
            test_evaluated=False,
        )
        record = dict(status="running", protocol=protocol, attempts=[], runs=[],
                      exact_replay_checks={})

    stage = output / "app"
    if source_hash(stage) != protocol["source_sha256"]:
        raise RuntimeError("Frozen application source changed")
    if file_hash(protocol["base_config"]["manifest"]) != protocol["manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed")
    record["runs"] = _verify_completed_runs(record, protocol)
    record.update(status="running")
    record.pop("error", None)
    write_json(output / "study.json", record)
    print(f"Compression study directory: {output}", flush=True)

    try:
        plan = [(method, 0) for method in BASE_METHODS]
        plan.extend(("qsgd-llz", p) for p in protocol["llz_p_values"])
        for seed in protocol["seeds"]:
            for method, p in plan:
                label = llz_label(p) if method == "qsgd-llz" else method
                if any(r["config"]["seed"] == seed and run_label(r) == label
                       for r in record["runs"]):
                    continue
                attempt_number = 1 + sum(a["seed"] == seed and a["label"] == label
                                         for a in record["attempts"])
                run_root = output / f"seed-{seed}-{label}-attempt-{attempt_number}"
                run_root.mkdir()
                config = dict(protocol["base_config"], seed=seed, compression=method,
                              **{"llz-p": p},
                              **{"output-dir": run_root.as_posix()})
                attempt = dict(seed=seed, method=method, label=label, llz_p=p,
                               output=str(run_root), status="running")
                record["attempts"].append(attempt)
                write_json(output / "study.json", record)
                print(f"Starting seed={seed}, method={label}, rounds={protocol['rounds']}", flush=True)
                try:
                    if launch(config, root=root, app_dir=stage,
                              log_path=run_root / "terminal.log") != 0:
                        raise RuntimeError(f"Flower failed: inspect {run_root / 'terminal.log'}")
                    result = collect(run_root, config)
                    if result["metadata"]["source_sha256"] != protocol["source_sha256"]:
                        raise RuntimeError("Run source differs from frozen application")
                    record["runs"].append(result)
                    attempt["status"] = "completed"
                    if method == "qsgd-llz" and p == 0:
                        qsgd = next(r for r in record["runs"] if
                                     r["config"]["seed"] == seed and
                                     r["config"]["compression"] == "qsgd")
                        exact = compare_lossless_pair(qsgd, result)
                        record["exact_replay_checks"][str(seed)] = dict(
                            rounds=exact["rounds"], status="passed",
                            upload_reduction_percent=exact["upload_reduction_percent"],
                            total_reduction_percent=exact["total_reduction_percent"])
                        print(f"Exact QSGD/LLZ p=0 replay passed: seed={seed}", flush=True)
                except BaseException as error:
                    attempt.update(status="failed", error=str(error))
                    raise
                write_json(output / "study.json", record)
                print(f"Completed seed={seed}, method={label}: "
                      f"{result['total_bytes']:,} serialized bytes", flush=True)

        record["summary"] = summarize(record["runs"], protocol["seeds"],
                                      protocol["llz_p_values"])
        record["status"] = "completed"
        write_json(output / "summary.json", record["summary"])
        write_json(output / "study.json", record)
        print(f"Completed compression study: {output / 'summary.json'}", flush=True)
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=str(error))
        write_json(output / "study.json", record)
        raise


if __name__ == "__main__":
    main()
