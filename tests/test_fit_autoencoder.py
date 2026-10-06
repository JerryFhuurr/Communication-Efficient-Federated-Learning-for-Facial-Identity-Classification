import numpy as np
import pytest
import torch

from compression import autoencoder as ae
from compression import fit_autoencoder as fit


def data():
    x = torch.zeros(4, 256)
    x[:, 0] = 1
    return dict(x=x, scales=torch.tensor([1., 2., 0., 1.]),
                weights=torch.tensor([1., 2., 3., 4.], dtype=torch.float64),
                mask=torch.arange(256)[None, :] < torch.tensor([1, 2, 3, 4])[:, None])


def test_weighted_loss_masks_padding_and_bypasses_zero_norm():
    d = data()
    prediction = torch.ones(4, 256)
    errors = fit.squared_errors(prediction, d, slice(None))
    torch.testing.assert_close(errors, torch.tensor([0., 1., 1., 3.]))
    # Zero input block is canonical: its original entries must also be zero.
    d['x'][2] = 0
    errors = fit.squared_errors(prediction, d, slice(None))
    torch.testing.assert_close(errors, torch.tensor([0., 1., 0., 3.]))
    class Ones(torch.nn.Module):
        def forward(self, x): return torch.ones_like(x)
    assert fit.validation_loss(Ones(), d) == pytest.approx(14 / 30)


def test_fit_selection_reuse_and_decoder_integrity(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(fit.CONFIG, 'epochs', 2)
    d = data(); d['x'][2] = 0
    model, record = fit.fit_client(0, d, d, tmp_path, 'test')
    assert record['selected_validation_mse'] == min(r['validation_normalized_masked_mse'] for r in record['history'])
    assert fit.validation_loss(model, d) == pytest.approx(record['selected_validation_mse'])
    original = ae.weight_bytes(model)
    restored, reused = fit.fit_client(0, d, d, tmp_path, 'test')
    assert reused == record and ae.weight_bytes(restored) == original
    path = tmp_path / 'client-0/decoder.aew'
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError): fit.fit_client(0, d, d, tmp_path, 'test')


def test_comparators_use_same_coordinates_and_count_real_bundle_bytes(monkeypatch):
    monkeypatch.setitem(fit.CONFIG, 'topk_k', 2)
    tensors = {'z': np.array([1., 3.], np.float32), 'a': np.array([2.], np.float32)}
    model = ae.BlockAutoencoder().eval()
    row = fit.probe(tensors, model, 'topk', 42)
    _, layout = ae._layout({k: v.shape for k, v in tensors.items()})
    assert row['packet_bytes'] == 8 + len(layout) + 12 + 16
    assert row['relative_squared_error'] == pytest.approx(1 / 14)
    assert row['energy'] == 14
    first = fit.probe(tensors, model, 'qsgd', 42)
    second = fit.probe(tensors, model, 'qsgd', 42)
    assert first['packet_sha256'] == second['packet_sha256']
    assert first['squared_error'] == second['squared_error']
    assert first['energy'] == row['energy']


def test_artifact_and_samples_checks(tmp_path):
    path = tmp_path / 'samples.npz'
    np.savez(path, blocks=np.zeros((1, 256), np.float32), scales=np.zeros(1, np.float32),
             population_weights=np.ones(1), valid_length=np.array([1], np.int32))
    entry = dict(path=path.name, bytes=path.stat().st_size, sha256=fit.file_hash(path))
    loaded = fit.read_samples(tmp_path, {'sampled_file': entry})
    assert loaded['mask'].sum() == 1
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError): fit.verified_path(tmp_path, entry)
