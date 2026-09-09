from copy import deepcopy

import pytest

from flower_face.compression_study import summarize


def run(seed, method, *, total, upload, accuracy=0.5, loss=1.0):
    config = {
        "seed": seed, "compression": method, "output-dir": f"{seed}-{method}",
        "qsgd-levels": 127, "llz-p": 0, "llz-window": 128,
    }
    metadata = {key: "same" for key in (
        "manifest_sha256", "source_sha256", "versions", "platform",
        "reproducibility", "initial_model_sha256", "selection_rule",
        "communication_measurement",
    )}
    replay = [dict(round=1, model_sha256=f"model-{seed}",
                   train={"train_loss": 2.0},
                   validation={"eval_loss": loss, "eval_acc": accuracy})]
    return dict(
        config=config, metadata=metadata, replay=replay,
        best={"step": 1, "validation_loss": loss, "validation_accuracy": accuracy},
        final_validation={"eval_loss": loss, "eval_acc": accuracy},
        output=f"output-{seed}-{method}", total_bytes=total,
        train_upload_bytes=upload,
        communication={"total": {"serialized_object_bytes": total}},
    )


def matrix():
    rows = []
    for seed in (42, 43):
        rows.append(run(seed, "none", total=1000, upload=800, accuracy=0.55, loss=0.9))
        qsgd = run(seed, "qsgd", total=700, upload=400)
        llz = deepcopy(qsgd)
        llz["config"]["compression"] = "qsgd-llz"
        llz["config"]["output-dir"] = f"{seed}-qsgd-llz"
        llz.update(output=f"output-{seed}-qsgd-llz", total_bytes=600,
                   train_upload_bytes=200,
                   communication={"total": {"serialized_object_bytes": 600}})
        rows.extend((qsgd, llz))
    return rows


def test_summary_requires_exact_lossless_pairs_and_reports_all_boundaries():
    result = summarize(matrix(), [42, 43])
    assert result["exact_qsgd_llz_replay"]
    assert set(result["methods"]) == {"none", "qsgd", "qsgd-llz"}
    stats = result["paired_statistics"]
    assert stats["qsgd_upload_reduction_vs_none_percent"]["mean"] == 50
    assert stats["llz_upload_reduction_vs_none_percent"]["mean"] == 75
    assert stats["llz_upload_reduction_vs_qsgd_percent"]["mean"] == 50
    assert stats["llz_total_reduction_vs_qsgd_percent"]["mean"] == pytest.approx(100 / 7)
    assert result["test_evaluated"] is False


def test_summary_rejects_missing_or_duplicate_matrix_entries():
    rows = matrix()
    with pytest.raises(RuntimeError, match="exactly one"):
        summarize(rows[:-1], [42, 43])
    with pytest.raises(RuntimeError, match="exactly one"):
        summarize(rows + [deepcopy(rows[0])], [42, 43])


@pytest.mark.parametrize("field", ["replay", "best"])
def test_summary_rejects_nonidentical_qsgd_llz_learning(field):
    rows = matrix()
    llz = next(r for r in rows if r["config"]["seed"] == 42 and
               r["config"]["compression"] == "qsgd-llz")
    if field == "replay":
        llz["replay"][0]["model_sha256"] = "different"
    else:
        llz["best"]["step"] = 2
    with pytest.raises(RuntimeError, match="Lossless integration mismatch"):
        summarize(rows, [42, 43])
