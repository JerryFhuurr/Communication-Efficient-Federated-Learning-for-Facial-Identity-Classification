from copy import deepcopy
import pytest
from flower_face.complete_comparison import summarize
from test_compression_study import matrix


def rows():
    runs = [r for r in matrix((0,1)) if r['config']['seed']==42]
    for run in runs: run['metadata']['fixed_state_policy']='same'
    topk = deepcopy(runs[0]); topk['config'].update(compression='topk-ef', **{'topk-k':124000})
    for run in runs: run['config']['topk-k']=124000
    topk['train_upload_bytes']=190; topk['total_bytes']=590
    topk['best']['validation_accuracy']=0.54
    return runs+[topk]


def test_five_method_summary_marks_feedback_and_reuses_lossless_gate():
    result = summarize(rows())
    assert result['exact_p0_replay']['exact_replay']
    assert result['methods']['topk-ef']['feedback']
    assert not result['methods']['qsgd']['feedback']
    assert result['topk_vs_p0']['upload_budget_difference_percent']==pytest.approx(-5)


@pytest.mark.parametrize('change',['source','platform','python','configuration','missing','duplicate'])
def test_controls_reject_confounded_or_incomplete_comparisons(change):
    runs=rows()
    if change in ['source','platform','python']:
        key='source_sha256' if change=='source' else change
        runs[-1]['metadata'][key]='different'
    elif change=='configuration': runs[-1]['config']['local-epochs']=10
    elif change=='missing': runs.pop()
    else: runs.append(deepcopy(runs[-1]))
    with pytest.raises((ValueError,RuntimeError)): summarize(runs)
