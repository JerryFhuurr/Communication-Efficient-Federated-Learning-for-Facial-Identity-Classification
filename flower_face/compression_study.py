"""Run a resumable paired study of FedAvg, QSGD, and QSGD+LLZ-p=0."""

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


METHODS = ("none", "qsgd", "qsgd-llz")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stats(values):
    return dict(mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values) > 1 else None)


def summarize(runs, seeds):
    """Validate the complete paired matrix and calculate thesis-facing metrics."""
    observed = [(r["config"]["seed"], r["config"]["compression"]) for r in runs]
    expected = {(seed, method) for seed in seeds for method in METHODS}
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("Expected exactly one completed run per seed and compression method")

    paired = []
    lossless = []
    for seed in seeds:
        methods = {r["config"]["compression"]: r for r in runs
                   if r["config"]["seed"] == seed}
        base, qsgd, llz = (methods[name] for name in METHODS)
        check_pair(base, qsgd)
        check_pair(base, llz)
        exact = compare_lossless_pair(qsgd, llz)
        lossless.append(dict(seed=seed, **exact))
        paired.append(dict(
            seed=seed,
            outputs={name: methods[name]["output"] for name in METHODS},
            selected={name: methods[name]["best"] for name in METHODS},
            total_serialized_bytes={name: methods[name]["total_bytes"] for name in METHODS},
            training_upload_bytes={name: methods[name]["train_upload_bytes"] for name in METHODS},
            qsgd_upload_reduction_vs_none_percent=
                100 * (1 - qsgd["train_upload_bytes"] / base["train_upload_bytes"]),
            llz_upload_reduction_vs_none_percent=
                100 * (1 - llz["train_upload_bytes"] / base["train_upload_bytes"]),
            llz_upload_reduction_vs_qsgd_percent=exact["upload_reduction_percent"],
            qsgd_total_reduction_vs_none_percent=
                100 * (1 - qsgd["total_bytes"] / base["total_bytes"]),
            llz_total_reduction_vs_none_percent=
                100 * (1 - llz["total_bytes"] / base["total_bytes"]),
            llz_total_reduction_vs_qsgd_percent=exact["total_reduction_percent"],
            qsgd_selected_accuracy_difference_pp=
                100 * (qsgd["best"]["validation_accuracy"] - base["best"]["validation_accuracy"]),
            qsgd_selected_loss_difference=
                qsgd["best"]["validation_loss"] - base["best"]["validation_loss"],
        ))

    methods = {}
    for method in METHODS:
        selected = [r for r in runs if r["config"]["compression"] == method]
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
        )

    reduction_keys = (
        "qsgd_upload_reduction_vs_none_percent",
        "llz_upload_reduction_vs_none_percent",
        "llz_upload_reduction_vs_qsgd_percent",
        "qsgd_total_reduction_vs_none_percent",
        "llz_total_reduction_vs_none_percent",
        "llz_total_reduction_vs_qsgd_percent",
        "qsgd_selected_accuracy_difference_pp",
        "qsgd_selected_loss_difference",
    )
    return dict(
        methods=methods,
        paired_seeds=paired,
        paired_statistics={key: _stats([row[key] for row in paired]) for key in reduction_keys},
        lossless_qsgd_llz=lossless,
        exact_qsgd_llz_replay=True,
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

    root = Path(__file__).resolve().parents[1]
    if args.resume:
        output = args.resume.resolve()
        import json
        record = json.loads((output / "study.json").read_text(encoding="utf-8"))
        protocol = record["protocol"]
        if (args.seeds != protocol["seeds"] or args.rounds != protocol["rounds"]
                or args.qsgd_levels != protocol["qsgd_levels"]
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
        compression_settings(dict(config, compression="qsgd-llz"))
        protocol = dict(
            seeds=args.seeds, rounds=args.rounds, methods=list(METHODS),
            qsgd_levels=args.qsgd_levels, llz_p=0, llz_window=args.llz_window,
            base_config=config, source_sha256=source_hash(stage),
            manifest_sha256=file_hash(config["manifest"]),
            framework=("Client model deltas; QSGD uses one L2 norm per tensor; LLZ-p=0 "
                       "encodes the identical QSGD codes; downlink is uncompressed."),
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
        for seed in protocol["seeds"]:
            for method in METHODS:
                if any(r["config"]["seed"] == seed and
                       r["config"]["compression"] == method for r in record["runs"]):
                    continue
                attempt_number = 1 + sum(a["seed"] == seed and a["method"] == method
                                         for a in record["attempts"])
                run_root = output / f"seed-{seed}-{method}-attempt-{attempt_number}"
                run_root.mkdir()
                config = dict(protocol["base_config"], seed=seed, compression=method,
                              **{"output-dir": run_root.as_posix()})
                attempt = dict(seed=seed, method=method, output=str(run_root), status="running")
                record["attempts"].append(attempt)
                write_json(output / "study.json", record)
                print(f"Starting seed={seed}, method={method}, rounds={protocol['rounds']}", flush=True)
                try:
                    if launch(config, root=root, app_dir=stage,
                              log_path=run_root / "terminal.log") != 0:
                        raise RuntimeError(f"Flower failed: inspect {run_root / 'terminal.log'}")
                    result = collect(run_root, config)
                    if result["metadata"]["source_sha256"] != protocol["source_sha256"]:
                        raise RuntimeError("Run source differs from frozen application")
                    record["runs"].append(result)
                    attempt["status"] = "completed"
                    if method == "qsgd-llz":
                        qsgd = next(r for r in record["runs"] if
                                     r["config"]["seed"] == seed and
                                     r["config"]["compression"] == "qsgd")
                        exact = compare_lossless_pair(qsgd, result)
                        record["exact_replay_checks"][str(seed)] = dict(
                            rounds=exact["rounds"], status="passed",
                            upload_reduction_percent=exact["upload_reduction_percent"],
                            total_reduction_percent=exact["total_reduction_percent"])
                        print(f"Exact QSGD/LLZ replay passed: seed={seed}", flush=True)
                except BaseException as error:
                    attempt.update(status="failed", error=str(error))
                    raise
                write_json(output / "study.json", record)
                print(f"Completed seed={seed}, method={method}: "
                      f"{result['total_bytes']:,} serialized bytes", flush=True)

        record["summary"] = summarize(record["runs"], protocol["seeds"])
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
