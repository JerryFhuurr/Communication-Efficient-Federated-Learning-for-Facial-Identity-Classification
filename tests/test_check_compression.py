from copy import deepcopy
import pytest

from flower_face.check_compression import summarize
from test_compression_study import matrix


def rows():
    return [r for r in matrix((0,1)) if r['config']['seed'] == 42 and r['config']['compression'] != 'none']


def test_three_method_summary_and_exact_replay():
    result = summarize(rows())
    assert result['exact_p0_replay']['exact_replay']
    assert result['methods']['qsgd-llz-p0']['upload_reduction_vs_qsgd_percent'] == 50
    assert result['methods']['qsgd-llz-p1']['selected_accuracy_difference_vs_qsgd_pp'] == pytest.approx(-5)


@pytest.mark.parametrize('damage',['missing','duplicate','model','platform','fixed','settings'])
def test_reject_confounds_and_incomplete_results(damage):
    runs = deepcopy(rows())
    if damage == 'missing': runs.pop()
    elif damage == 'duplicate': runs[2] = deepcopy(runs[1])
    elif damage == 'model': runs[1]['replay'][0]['model_sha256'] = 'wrong'
    elif damage == 'platform': runs[2]['metadata']['platform'] = 'different'
    elif damage == 'fixed': runs[2]['metadata']['fixed_state_policy'] = 'different'
    elif damage == 'settings': runs[2]['config']['learning-rate'] = 10
    with pytest.raises((ValueError,RuntimeError)): summarize(runs)
