import struct

import numpy as np
import pytest

from compression.qsgd import (HEADER, QuantizedTensor, decode, dequantize, encode,
                              pack, packet_stats, quantize, unpack)


def test_known_grid_values_have_exact_codes_and_known_wire_bytes():
    x = np.array([3, -4, 0], dtype=np.float32)
    q = quantize(x, levels=5, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(q.codes, [3, -4, 0])
    assert q.norm == 5
    # 16-byte header + uint32 shape + symbols 0011 1100 0000 + 4 padding bits.
    expected = struct.pack("<4sBBHdI", b"QSD1", 1, 1, 5, 5.0, 3) + bytes.fromhex("3c00")
    assert pack(q) == expected
    np.testing.assert_array_equal(decode(expected), x)
    stats = packet_stats(expected)
    assert stats['metadata_bytes'] == 20
    assert stats['code_bits'] == 12 and stats['padding_bits'] == 4
    assert stats['encoded_bytes'] == 22 and stats['encoded_bits'] == 176


@pytest.mark.parametrize('levels', [1, 2, 3, 7, 127, 128, 255, 65535])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_endpoints_shapes_and_integer_codes_survive_packing(levels, dtype):
    x = np.array([[0, 2, -3], [4, 0, -7]], dtype=dtype).T  # noncontiguous
    q = quantize(x, levels=levels, rng=np.random.default_rng(7))
    restored = unpack(pack(q))
    np.testing.assert_array_equal(restored.codes, q.codes)
    assert restored.norm == q.norm and restored.dtype == q.dtype and restored.levels == q.levels
    decoded = decode(pack(q))
    assert decoded.dtype == x.dtype and decoded.shape == x.shape
    np.testing.assert_array_equal(decoded, dequantize(q))
    assert np.all(decoded[x == 0] == 0)
    assert np.all(decoded[x < 0] <= 0) and np.all(decoded[x > 0] >= 0)
    endpoint = quantize(np.array([-2.0], dtype=dtype), levels=levels, rng=np.random.default_rng(1))
    assert endpoint.codes[0] == -levels  # the magnitude endpoint s must fit
    np.testing.assert_array_equal(decode(pack(endpoint)), [-2.0])


@pytest.mark.parametrize('shape', [(), (0,), (2, 0, 3), (3, 4)])
def test_zero_and_empty_tensors(shape):
    x = np.zeros(shape, dtype=np.float32)
    packet = encode(x, levels=127, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(decode(packet), x)
    assert decode(packet).shape == shape and unpack(packet).norm == 0


def test_seed_reproducibility_rng_advancement_and_no_input_mutation():
    x = np.random.default_rng(1).normal(size=512).astype(np.float32)
    original = x.copy()
    rng = np.random.default_rng(42)
    first = encode(x, levels=7, rng=rng)
    assert first == encode(x, levels=7, rng=np.random.default_rng(42))
    assert first != encode(x, levels=7, rng=rng)
    np.testing.assert_array_equal(x, original)


def test_stochastic_rounding_mean_and_variance_match_analytical_values():
    x = np.array([0.1, -0.4, 0.7, -1.2, 0], dtype=np.float64)
    levels, trials = 3, 12000
    rng = np.random.default_rng(123)
    samples = np.array([decode(encode(x, levels=levels, rng=rng)) for _ in range(trials)])
    norm = np.linalg.norm(x)
    scaled = np.abs(x) / norm * levels
    probability = scaled - np.floor(scaled)
    variance = (norm / levels) ** 2 * probability * (1 - probability)
    # A six-standard-error check, not a claim that finite samples prove unbiasedness.
    assert np.all(np.abs(samples.mean(axis=0) - x) <= 6 * np.sqrt(variance / trials) + 1e-12)
    np.testing.assert_allclose(samples.var(axis=0), variance, rtol=0.08, atol=1e-12)
    mse = np.mean(np.sum((samples - x) ** 2, axis=1))
    bound = min(x.size / levels**2, np.sqrt(x.size) / levels) * norm**2
    assert mse <= 1.08 * bound


@pytest.mark.parametrize('value', [1e-300, 1e300])
def test_stable_norm_for_extreme_finite_values(value):
    x = np.array([value, -value], dtype=np.float64)
    q = quantize(x, levels=127, rng=np.random.default_rng(7))
    assert q.norm > 0 and np.isfinite(q.norm)
    assert np.isfinite(decode(pack(q))).all()


@pytest.mark.parametrize('levels', [0, -1, 65536, 2.5, True])
def test_invalid_levels(levels):
    with pytest.raises(ValueError, match='levels'):
        encode(np.ones(3, dtype=np.float32), levels=levels, rng=np.random.default_rng(1))


@pytest.mark.parametrize('x', [np.array([np.nan]), np.array([np.inf]), np.array([1], dtype=np.int32),
                             np.array([1], dtype=np.float16), np.array([1j])])
def test_invalid_arrays(x):
    with pytest.raises(ValueError):
        encode(x, levels=7, rng=np.random.default_rng(1))


def test_reject_malformed_packets():
    header = HEADER.pack(b'QSD1', 1, 1, 5, 1.0) + struct.pack('<I', 1)
    valid = header + b'\x50'  # positive magnitude 5, four padding zeros
    np.testing.assert_array_equal(decode(valid), [1.0])
    malformed = [b'', valid[:10], valid[:18], valid[:-1], valid + b'\0',
                 b'BAD1' + valid[4:], valid[:4] + b'\x09' + valid[5:],
                 valid[:5] + b'\xff' + valid[6:], valid[:6] + b'\0\0' + valid[8:],
                 header + b'\x51',  # nonzero padding
                 header + b'\x70',  # unused magnitude 7 when s=5
                 header + b'\x80',  # negative zero
                 HEADER.pack(b'QSD1', 1, 1, 5, float('nan')) + valid[16:],
                 HEADER.pack(b'QSD1', 1, 1, 5, 0) + valid[16:],
                 valid[:16] + struct.pack('<I', 2**32 - 1) + valid[20:]]
    for packet in malformed:
        with pytest.raises(ValueError):
            decode(packet)


def test_packet_size_includes_all_metadata_and_padding():
    x = np.random.default_rng(1).normal(size=4096).astype(np.float32)
    payload = encode(x, levels=127, rng=np.random.default_rng(42))
    stats = packet_stats(payload)
    assert stats['raw_bytes'] == 16384
    assert stats['encoded_bytes'] == 4116  # 4096 one-byte symbols + 20-byte header/shape
    assert stats['encoded_bits'] == 8*stats['metadata_bytes'] + stats['code_bits'] + stats['padding_bits']


def test_invalid_quantized_state_cannot_silently_truncate_codes():
    for codes in [np.array([1.5]), np.array([8]), np.array([-8])]:
        with pytest.raises(ValueError):
            pack(QuantizedTensor(codes, 1.0, 7, 'float32'))
    with pytest.raises(ValueError, match='range'):
        dequantize(QuantizedTensor(np.array([1]), 1e39, 1, 'float32'))
    with pytest.raises(ValueError, match='range'):
        pack(QuantizedTensor(np.array([1]), 1e39, 1, 'float32'))
