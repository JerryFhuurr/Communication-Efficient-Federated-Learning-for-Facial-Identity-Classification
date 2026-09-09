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


def matrix(p_values=(0,)):
    rows = []
    for seed in (42, 43):
        rows.append(run(seed, "none", total=1000, upload=800, accuracy=0.55, loss=0.9))
        qsgd = run(seed, "qsgd", total=700, upload=400)
        rows.append(qsgd)
        for p in p_values:
            llz = deepcopy(qsgd)
            llz["config"].update({"compression": "qsgd-llz", "llz-p": p,
                                  "output-dir": f"{seed}-qsgd-llz-p{p}"})
            llz.update(output=f"output-{seed}-qsgd-llz-p{p}", total_bytes=600-p*50,
                       train_upload_bytes=200-p*50,
                       communication={"total": {"serialized_object_bytes": 600-p*50}})
            if p:
                llz["replay"][0].update(model_sha256=f"lossy-{seed}")
                llz["replay"][0]["validation"].update(eval_loss=1.1, eval_acc=0.45)
                llz["replay"][0]["train"].update(
                    qsgd_relative_squared_error=0.1,
                    llz_relative_squared_error=0.2)
                llz["best"].update(validation_loss=1.1, validation_accuracy=0.45)
                llz["final_validation"].update(eval_loss=1.1, eval_acc=0.45)
            rows.append(llz)
    return rows


def test_summary_requires_exact_lossless_pairs_and_reports_all_boundaries():
    result = summarize(matrix(), [42, 43])
    assert result["exact_qsgd_llz_p0_replay"]
    assert set(result["methods"]) == {"none", "qsgd", "qsgd-llz-p0"}
    stats = result["paired_statistics"]
    assert stats["qsgd_upload_reduction_vs_none_percent"]["mean"] == 50
    llz = result["llz_tolerance_statistics"]["qsgd-llz-p0"]
    assert llz["upload_reduction_vs_none_percent"]["mean"] == 75
    assert llz["upload_reduction_vs_qsgd_percent"]["mean"] == 50
    assert llz["total_reduction_vs_qsgd_percent"]["mean"] == pytest.approx(100 / 7)
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


def test_summary_accepts_lossy_tolerance_and_keeps_its_learning_difference():
    result = summarize(matrix((0, 1)), [42, 43], (0, 1))
    assert set(result["methods"]) == {
        "none", "qsgd", "qsgd-llz-p0", "qsgd-llz-p1"}
    lossy = result["llz_tolerance_statistics"]["qsgd-llz-p1"]
    assert lossy["exact_qsgd_replay"] is False
    assert lossy["selected_accuracy_difference_vs_qsgd_pp"]["mean"] == pytest.approx(-5)
    assert lossy["mean_round_qsgd_relative_squared_error"]["mean"] == 0.1
    assert lossy["mean_round_llz_relative_squared_error"]["mean"] == 0.2


def test_summary_rejects_lossy_training_confound():
    rows = matrix((0, 1))
    lossy = next(r for r in rows if r["config"]["compression"] == "qsgd-llz"
                 and r["config"]["llz-p"] == 1)
    lossy["config"]["llz-window"] = 64
    with pytest.raises(RuntimeError, match="confound"):
        summarize(rows, [42, 43], (0, 1))
