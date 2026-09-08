from copy import deepcopy

import pytest

from flower_face.check_llz import compare


def records():
    base = dict(config={'compression': 'qsgd', 'llz-p': 0, 'llz-window': 128, 'qsgd-levels': 127},
        metadata={key: 'same' for key in ('manifest_sha256', 'source_sha256', 'versions', 'platform',
            'reproducibility', 'initial_model_sha256', 'selection_rule', 'communication_measurement')},
        replay=[dict(round=1, model_sha256='model', train={'loss': 1}, validation={'loss': 2})],
        best={'step': 1}, output='base', communication={'total': {'serialized_object_bytes': 1000}},
        total_bytes=1000, train_upload_bytes=400)
    other = deepcopy(base)
    other['config']['compression'] = 'qsgd-llz'
    other.update(output='llz', total_bytes=700, train_upload_bytes=100,
                 communication={'total': {'serialized_object_bytes': 700}})
    return base, other


def test_exact_learning_comparison_excludes_communication_variation():
    report = compare(*records())
    assert report['exact_replay'] and report['rounds'] == 1
    assert report['upload_reduction_percent'] == 75
    assert report['total_reduction_percent'] == pytest.approx(30)


@pytest.mark.parametrize('difference', ['model', 'train', 'validation', 'best', 'source', 'measurement', 'window', 'lossy', 'method'])
def test_comparison_rejects_unmatched_runs(difference):
    base, other = records()
    if difference in ('model', 'train', 'validation'):
        other['replay'][0]['model_sha256' if difference == 'model' else difference] = 'changed'
    elif difference == 'best':
        other['best']['step'] = 2
    elif difference in ('source', 'measurement'):
        other['metadata']['source_sha256' if difference == 'source' else 'communication_measurement'] = 'changed'
    else:
        key, value = {'window': ('llz-window', 64), 'lossy': ('llz-p', 1), 'method': ('compression', 'none')}[difference]
        other['config'][key] = value
    with pytest.raises((RuntimeError, ValueError)):
        compare(base, other)
