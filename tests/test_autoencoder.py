"""Protocol correctness only; random weights say nothing about codec quality."""
import hashlib
import struct

import numpy as np
import pytest
import torch

from compression import autoencoder as ae
from compression.block_data import normalize_blocks, padded_blocks


@pytest.fixture
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return ae.BlockAutoencoder().eval()


def test_packet_matches_reference_forward_and_crops_each_tensor(model):
    tensors = {'z': np.arange(257, dtype=np.float32),
               'a': np.array([[3., 4.], [0., 0.]], np.float32),
               'zero': np.zeros(259, np.float32), 'scalar': np.array(2., np.float32)}
    before = ae.weight_bytes(model)
    rng = torch.get_rng_state().clone()
    packet = ae.encode(tensors, model)
    shapes = {k: v.shape for k, v in tensors.items()}
    result = ae.decode(packet, model.decoder, shapes)
    assert list(result) == sorted(tensors)
    for name, original in tensors.items():
        blocks, scales = normalize_blocks(padded_blocks(original))
        with torch.inference_mode():
            expected = model(torch.from_numpy(blocks)).numpy() * scales[:, None]
        expected[scales == 0] = 0
        np.testing.assert_allclose(result[name], expected.reshape(-1)[:original.size].reshape(original.shape), rtol=1e-6, atol=1e-7)
        assert result[name].dtype == np.float32
    np.testing.assert_array_equal(result['zero'], tensors['zero'])
    assert ae.encode(dict(reversed(list(tensors.items()))), model) == packet
    assert ae.weight_bytes(model) == before
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    _, header = ae._layout(shapes)
    blocks = sum((v.size + 255) // 256 for v in tensors.values())
    assert len(packet) == 44 + len(header) + 28 * blocks


def test_weights_roundtrip_and_decoder_distribution(tmp_path, model):
    data = {'x': np.ones(300, np.float32)}
    packet = ae.encode(data, model)
    for decoder_only in [False, True]:
        artifact = ae.weight_bytes(model, decoder_only=decoder_only)
        path = tmp_path / ('decoder.aew' if decoder_only else 'model.aew')
        path.write_bytes(artifact)
        restored = ae.load_weights(path.read_bytes(), decoder_only=decoder_only)
        if decoder_only:
            assert len(artifact) == 68352 + 37
            np.testing.assert_array_equal(ae.decode(packet, restored, {'x': (300,)})['x'],
                                          ae.decode(packet, model.decoder, {'x': (300,)})['x'])
        else:
            assert ae.weight_bytes(restored) == artifact
            assert ae.encode(data, restored) == packet
        with pytest.raises(ValueError): ae.load_weights(artifact + b'x', decoder_only=decoder_only)
        damaged = bytearray(artifact); damaged[20] ^= 1
        with pytest.raises(ValueError): ae.load_weights(damaged, decoder_only=decoder_only)
        with pytest.raises(ValueError): ae.load_weights(artifact, decoder_only=not decoder_only)


@pytest.mark.parametrize('damage', ['short', 'trailing', 'version', 'count', 'layout', 'hash', 'nan', 'negative', 'zero-latent'])
def test_malformed_packets_fail_closed(model, damage):
    packet = bytearray(ae.encode({'x': np.ones(256, np.float32)}, model))
    payload_start = 44 + struct.unpack_from('<I', packet, 4)[0]
    if damage == 'short': packet = packet[:-1]
    elif damage == 'trailing': packet += b'x'
    elif damage == 'version': packet[3] = ord('2')
    elif damage == 'count': struct.pack_into('<I', packet, 8, 999)
    elif damage == 'layout': packet[46] ^= 1
    elif damage == 'hash': packet[12] ^= 1
    elif damage == 'nan': struct.pack_into('<f', packet, payload_start + 4, float('nan'))
    elif damage == 'negative': struct.pack_into('<f', packet, payload_start, -1.)
    elif damage == 'zero-latent':
        struct.pack_into('<f', packet, payload_start, 0.)
        struct.pack_into('<f', packet, payload_start + 4, 1.)
    with pytest.raises(ValueError): ae.decode(bytes(packet), model.decoder, {'x': (256,)})


def test_wrong_decoder_layout_architecture_and_training_mode(model):
    packet = ae.encode({'x': np.ones(256, np.float32)}, model)
    with pytest.raises(ValueError): ae.decode(packet, model.decoder, {'x': (16, 16)})
    other = ae.load_weights(ae.weight_bytes(model))
    with torch.no_grad(): other.decoder[2].bias[0] += 1
    with pytest.raises(ValueError): ae.decode(packet, other.decoder, {'x': (256,)})
    other = ae.load_weights(ae.weight_bytes(model))
    other.decoder[1] = torch.nn.Sigmoid()
    with pytest.raises(ValueError): ae.encode({'x': np.ones(256, np.float32)}, other)
    model.train()
    with pytest.raises(ValueError): ae.encode({'x': np.ones(256, np.float32)}, model)
    with pytest.raises(ValueError): ae.decode(packet, model.decoder, {'x': (256,)})


@pytest.mark.parametrize('values', [np.ones(3, np.float64), np.array([np.nan], np.float32), np.zeros(0, np.float32)])
def test_invalid_input(model, values):
    with pytest.raises(ValueError): ae.encode({'x': values}, model)


def test_chunk_limit_and_nonfinite_output(model):
    calls = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
    ae.encode({'x': np.ones(256 * 1025, np.float32)}, model)
    hook.remove()
    assert calls == [1024, 1]
    with torch.no_grad(): model.decoder[2].bias.fill_(torch.finfo(torch.float32).max)
    packet = ae.encode({'x': np.full(256, 2., np.float32)}, model)
    with pytest.raises(ValueError, match='Nonfinite'): ae.decode(packet, model.decoder, {'x': (256,)})


def test_nonfinite_weights_rejected_even_with_recomputed_checksum(model):
    payload = bytearray(ae.weight_bytes(model))
    struct.pack_into('<f', payload, 5, float('nan'))
    payload[-32:] = hashlib.sha256(payload[:-32]).digest()
    with pytest.raises(ValueError): ae.load_weights(bytes(payload))
