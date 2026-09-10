from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flower_face.task import train, validate_weight_decay


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
