from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flower_face.task import train, validate_weight_decay
from scripts.sweep_weight_decay import check_control, summarize


def test_weight_decay_matches_sgd_equation_for_weights_and_biases():
    model = nn.Linear(2, 2).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.2, -.4], [.6, .3]]))
        model.bias.copy_(torch.tensor([.1, -.2]))
    images = torch.tensor([[1., 2.], [-.5, .8]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    initial = deepcopy(model.state_dict())
    loss = nn.functional.cross_entropy(model(images), labels)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    reported = train(model, [(images, labels)], 1, .1, weight_decay=.03)
    assert reported == loss.item()  # Still cross-entropy, not a penalized metric.
    for (name, value), gradient in zip(model.named_parameters(), gradients):
        expected = initial[name] - .1 * (gradient + .03 * initial[name])
        torch.testing.assert_close(value, expected, rtol=1e-14, atol=1e-14)


def test_zero_decay_is_bitwise_equal_to_legacy_sgd_over_multiple_epochs():
    generator = torch.Generator().manual_seed(57)
    dataset = TensorDataset(torch.randn(13, 3, generator=generator), torch.arange(13) % 2)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = nn.Linear(3, 2)
    legacy = deepcopy(model)
    optimizer = torch.optim.SGD(legacy.parameters(), lr=.1)
    total, count = 0., 0
    for _ in range(3):
        for images, labels in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(legacy(images), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(labels)
            count += len(labels)
    assert train(model, loader, 3, .1, weight_decay=0) == total/count
    for name, value in model.state_dict().items():
        assert torch.equal(value, legacy.state_dict()[name])


@pytest.mark.parametrize('value', [-.01, float('nan'), float('inf'), -float('inf'), True, '0.1', None])
def test_invalid_weight_decay_is_rejected(value):
    with pytest.raises(ValueError, match='weight-decay'):
        validate_weight_decay(value)


def result(seed=42, decay=0., loss=2.):
    return dict(config={'seed':seed, 'weight-decay':decay, 'learning-rate':.1, 'compression':'none',
                        'evaluate-final-test':False, 'output-dir':'example'},
        metadata=dict.fromkeys(('manifest_sha256','initial_model_sha256','versions','python','platform',
            'reproducibility','selection_rule','source_sha256','communication_measurement'), 'same'),
        replay=[dict(round=1, model_sha256='model', train={'loss':1.}, validation={'loss':2.})],
        best=dict(validation_loss=loss, validation_accuracy=.4),
        checkpoint_evaluation={'final':{'train':{'accuracy':1.}}})


def test_legacy_bridge_allows_source_change_only_with_exact_learning_replay():
    old, new = result(), result()
    del old['config']['weight-decay']
    new['metadata']['source_sha256'] = 'weight-decay-support'
    new['metadata']['communication_measurement'] = 'updated-schema'
    check_control(old, new, legacy_replay=True)
    for field in ('model_sha256', 'train', 'validation'):
        changed = deepcopy(new)
        changed['replay'][0][field] = 'different'
        with pytest.raises(RuntimeError, match='replay'):
            check_control(old, changed, legacy_replay=True)
    new['config']['weight-decay'] = .001
    with pytest.raises(RuntimeError):
        check_control(old, new, legacy_replay=True)


def test_new_decay_runs_require_same_source_and_learning_rate():
    base, new = result(), result(decay=.001)
    check_control(base, new)
    new['metadata']['source_sha256'] = 'different'
    with pytest.raises(RuntimeError, match='source_sha256'):
        check_control(base, new)
    new = result(decay=.001)
    new['config']['learning-rate'] = .03
    with pytest.raises(RuntimeError, match='confound'):
        check_control(base, new)


def test_decay_selection_rejects_incomplete_runs_and_uses_mean_loss():
    runs = [result(seed, decay, loss=2.-decay*100) for seed in (42,43,44) for decay in (0.,.001)]
    assert summarize(runs,[42,43,44],[0.,.001])['preferred_weight_decay'] == .001
    with pytest.raises(RuntimeError):
        summarize(runs[:-1],[42,43,44],[0.,.001])
    for r in runs:
        r['best']['validation_loss'] = 2.
    assert summarize(runs,[42,43,44],[.001,0.])['preferred_weight_decay'] == 0.
