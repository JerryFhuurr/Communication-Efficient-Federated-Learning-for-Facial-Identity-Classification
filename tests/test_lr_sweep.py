from copy import deepcopy

import pytest

from scripts.sweep_learning_rate import check_control, summarize


def run(seed=42, rate=.01, loss=2., accuracy=.5):
    return dict(config={'seed': seed, 'learning-rate': rate, 'output-dir': 'example',
                        'compression': 'none', 'evaluate-final-test': False, 'local-epochs': 1},
                metadata=dict.fromkeys(('source_sha256', 'manifest_sha256', 'initial_model_sha256', 'versions',
                    'python', 'platform', 'reproducibility', 'selection_rule', 'communication_measurement'), 'same'),
                best=dict(validation_loss=loss, validation_accuracy=accuracy),
                checkpoint_evaluation={'final': {'train': {'accuracy': .75}}})


def test_learning_rate_and_output_are_allowed_but_training_confounds_are_rejected():
    base = run()
    candidate = run(rate=.03)
    candidate['config']['output-dir'] = 'new'
    check_control(base, candidate)
    for key, value in [('seed', 43), ('local-epochs', 2), ('compression', 'qsgd'), ('evaluate-final-test', True)]:
        changed = deepcopy(candidate)
        changed['config'][key] = value
        with pytest.raises(RuntimeError):
            check_control(base, changed)


@pytest.mark.parametrize('key', ['source_sha256', 'manifest_sha256', 'initial_model_sha256', 'versions',
    'python', 'platform', 'reproducibility', 'selection_rule', 'communication_measurement'])
def test_mismatched_experiment_controls_are_rejected(key):
    candidate = run(rate=.03)
    candidate['metadata'][key] = 'different'
    with pytest.raises(RuntimeError, match=key):
        check_control(run(), candidate)


def test_rate_selection_uses_mean_loss_not_accuracy_or_best_single_seed():
    runs = [run(seed, .01, loss=2., accuracy=.9) for seed in (42, 43, 44)]
    runs += [run(seed, .03, loss=loss, accuracy=.4) for seed, loss in zip((42, 43, 44), (1., 1.5, 3.))]
    result = summarize(runs, [42, 43, 44], [.01, .03])
    assert result['preferred_learning_rate'] == .03
    for r in runs:
        r['best']['validation_loss'] = 2.
    assert summarize(runs, [42, 43, 44], [.03, .01])['preferred_learning_rate'] == .01


def test_partial_duplicate_and_nonfinite_results_cannot_select_a_winner():
    runs = [run(seed, rate) for seed in (42, 43, 44) for rate in (.01, .03)]
    for invalid in (runs[:-1], runs + [runs[0]], runs[:-1] + [runs[0]]):
        with pytest.raises(RuntimeError):
            summarize(invalid, [42, 43, 44], [.01, .03])
    runs[0]['best']['validation_loss'] = float('nan')
    with pytest.raises(RuntimeError, match='Nonfinite'):
        summarize(runs, [42, 43, 44], [.01, .03])
