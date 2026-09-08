import itertools
import struct
import subprocess
import sys

import numpy as np
import pytest

from compression import llz_p, qsgd, qsgd_llz


def reference_parse(source, p, window):
    """Independent length-first exhaustive search from revised Algorithm 2."""
    decoded, tokens = [], []
    while len(decoded) < len(source):
        pointer = len(decoded)
        dictionary = decoded[-window:]
        best_length, best_position = 0, 0
        for length in range(1, min(len(dictionary), len(source)-pointer-1)+1):
            for start in range(len(dictionary)-length+1):
                if all(abs(int(source[pointer+j])-dictionary[start+j]) <= p for j in range(length)):
                    best_length, best_position = length, start+1
        literal = int(source[pointer+best_length])
        decoded += dictionary[best_position-1:best_position-1+best_length] if best_length else []
        decoded.append(literal)
        tokens.append((best_position, best_length, literal))
    return tokens, decoded


def packet_from_tokens(tokens, *, shape, bound=2, p=0, window=5, maximum=None):
    """Independent bit-string encoder for known answers/malformed packet tests."""
    if maximum is None:
        maximum = max((t[1] for t in tokens), default=0)
    widths = (window.bit_length(), maximum.bit_length(), (2*bound).bit_length())
    bits = ''.join(format(v, f'0{width}b') for token in tokens
                   for v, width in zip((token[0], token[1], token[2]+bound), widths) if width)
    padded = bits+'0'*((-len(bits)) % 8)
    body = bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))
    return (struct.pack('<4sBHIIII', b'LLP1', len(shape), bound, p, window, maximum, len(tokens))
            + struct.pack('<'+'I'*len(shape), *shape)+body)


def test_manuscript_worked_example_tokens_reconstruction_and_known_bytes():
    x = np.array([1, 1, -1, 2, -1, 0, 2, -1, -1, 0], dtype=np.int32)
    tokens = [(0, 0, 1), (1, 1, -1), (2, 2, 0), (3, 3, 0)]
    assert llz_p.parse(x, alphabet_bound=2, p=1, window_size=5) == tokens
    packet = llz_p.encode(x, alphabet_bound=2, p=1, window_size=5)
    assert packet == packet_from_tokens(tokens, shape=(10,), p=1)
    np.testing.assert_array_equal(llz_p.decode(packet), [1, 1, -1, 1, -1, 0, 1, -1, 0, 0])
    stats = llz_p.packet_stats(packet)
    assert (stats['position_bits'], stats['length_bits'], stats['literal_bits']) == (3, 2, 3)
    assert stats['triplet_bits'] == 32 and stats['metadata_bytes'] == 27
    assert stats['encoded_bytes'] == 31 and stats['padding_bits'] == 0


def test_exhaustive_small_sequences_agree_with_independent_reference():
    for size in range(7):
        for sequence in itertools.product((-1, 0, 1), repeat=size):
            x = np.array(sequence, dtype=np.int8)
            for p, window in itertools.product((0, 1), (1, 3)):
                tokens, expected = reference_parse(sequence, p, window)
                assert llz_p.parse(x, alphabet_bound=1, p=p, window_size=window) == tokens
                decoded = llz_p.decode(llz_p.encode(x, alphabet_bound=1, p=p, window_size=window))
                np.testing.assert_array_equal(decoded, expected)
                assert np.all(np.abs(decoded-x.astype(np.int64)) <= p)


@pytest.mark.parametrize('bound', [1, 4, 127, 128, 65535])
@pytest.mark.parametrize('shape', [(), (0,), (2, 0, 3), (3, 19)])
def test_lossless_roundtrip_shapes_endpoints_noncontiguous_and_no_mutation(bound, shape):
    rng = np.random.default_rng(42)
    x = rng.integers(-bound, bound+1, size=shape, dtype=np.int64)
    if x.ndim == 2 and x.size:
        x[0, 0], x[0, 1] = -bound, bound
        x = x.T
    before = x.copy()
    payload = llz_p.encode(x, alphabet_bound=bound)
    out = llz_p.decode(payload)
    np.testing.assert_array_equal(out, x)
    np.testing.assert_array_equal(x, before)
    assert out.shape == x.shape and out.dtype == np.int32
    assert payload == llz_p.encode(x, alphabet_bound=bound)


def test_reconstructed_history_prevents_cascading_distortion():
    x = np.repeat(np.arange(12, dtype=np.int32), 4)
    for p in [0, 1, 2]:
        payload = llz_p.encode(x, alphabet_bound=11, p=p, window_size=5)
        decoded = llz_p.decode(payload)
        _, expected = reference_parse(x, p, 5)
        np.testing.assert_array_equal(decoded, expected)
        assert np.max(np.abs(decoded-x)) <= p


def test_match_ties_window_sliding_and_reserved_final_literal():
    x = np.zeros(40, dtype=np.int32)
    tokens = llz_p.parse(x, alphabet_bound=1, window_size=3)
    assert tokens[:3] == [(0, 0, 0), (1, 1, 0), (1, 3, 0)]
    assert sum(length+1 for _, length, _ in tokens) == len(x)
    for p in (0, 1, 2):
        for window in (1, 2, 3, 8):
            rng = np.random.default_rng(71)
            x = rng.integers(-2, 3, 100, dtype=np.int32)
            assert llz_p.parse(x, alphabet_bound=2, p=p, window_size=window) == reference_parse(x, p, window)[0]


@pytest.mark.parametrize('kwargs', [dict(p=-1), dict(p=0.5), dict(p=True), dict(p=5),
    dict(window_size=0), dict(window_size=65536), dict(window_size=1.0),
    dict(alphabet_bound=0), dict(alphabet_bound=65536), dict(alphabet_bound=True)])
def test_invalid_settings(kwargs):
    settings = dict(alphabet_bound=2)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        llz_p.encode(np.array([1], dtype=np.int32), **settings)


@pytest.mark.parametrize('x', [np.array([1.0]), np.array([True]), np.array([1], dtype=np.uint32),
                              np.array([3]), np.array([-3]), np.array([2**63-1])])
def test_invalid_symbols(x):
    with pytest.raises(ValueError):
        llz_p.encode(x, alphabet_bound=2)


@pytest.mark.parametrize('tokens,shape,maximum', [
    ([(1, 1, 0)], (2,), 1),  # reference before any dictionary exists
    ([(1, 0, 0)], (1,), 0),  # noncanonical literal
    ([(0, 0, 0), (0, 1, 0)], (3,), 1),
    ([(0, 0, 0), (1, 2, 0)], (4,), 2),  # overlapping reference
    ([(0, 0, 0), (2, 1, 0)], (3,), 1),  # out-of-window start
    ([(0, 0, 3)], (1,), 0),  # unused alphabet bit pattern
    ([(0, 0, 0), (1, 1, 0)], (2,), 1),  # decoded extent exceeds shape
    ([(0, 0, 0), (1, 1, 0)], (4,), 1),  # too few decoded symbols
    ([(0, 0, 0), (1, 1, 0)], (3,), 2),  # unused declared maximum
])
def test_malformed_tokens(tokens, shape, maximum):
    with pytest.raises(ValueError):
        llz_p.decode(packet_from_tokens(tokens, shape=shape, maximum=maximum))


def test_truncation_padding_versions_shape_and_allocation_guard():
    packet = llz_p.encode(np.array([1], dtype=np.int32), alphabet_bound=2, window_size=5)
    for size in range(len(packet)):
        with pytest.raises(ValueError):
            llz_p.decode(packet[:size])
    for broken in [packet+b'\x00', b'BAD1'+packet[4:], packet[:-1]+bytes([packet[-1] | 1])]:
        with pytest.raises(ValueError):
            llz_p.decode(broken)
    with pytest.raises(ValueError, match='max_symbols'):
        llz_p.decode(packet, max_symbols=0)
    # A tiny hostile packet must be rejected before allocating its claimed shape.
    huge = bytearray(packet)
    struct.pack_into('<I', huge, llz_p.HEADER.size, 0xFFFFFFFF)
    with pytest.raises(ValueError, match='max_symbols'):
        llz_p.decode(bytes(huge))
    with pytest.raises(TypeError):
        llz_p.decode(bytearray(packet))


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('shape', [(), (0,), (2, 0, 3), (8, 32)])
def test_combined_p0_preserves_codes_metadata_and_float_bytes(dtype, shape):
    x = np.random.default_rng(2).normal(size=shape).astype(dtype)
    q = qsgd.quantize(x, levels=127, rng=np.random.default_rng(3))
    packet = qsgd_llz.pack(q)
    restored = qsgd_llz.unpack(packet)
    np.testing.assert_array_equal(restored.codes, q.codes)
    assert (restored.norm, restored.levels, restored.dtype) == (q.norm, q.levels, q.dtype)
    assert qsgd_llz.decode(packet).tobytes() == qsgd.decode(qsgd.pack(q)).tobytes()
    stats = qsgd_llz.packet_stats(packet)
    assert stats['encoded_bits'] == stats['metadata_bytes']*8+stats['triplet_bits']+stats['padding_bits']
    assert stats['encoded_bytes'] == len(packet)


def test_lossy_symbol_and_dequantized_error_bounds():
    x = np.random.default_rng(6).normal(size=512).astype(np.float64)
    q = qsgd.quantize(x, levels=7, rng=np.random.default_rng(8))
    for p in [0, 1, 2, 14]:
        restored = qsgd_llz.unpack(qsgd_llz.pack(q, p=p))
        difference = np.abs(restored.codes-q.codes)
        assert np.max(difference) <= p
        error = qsgd.dequantize(restored)-qsgd.dequantize(q)
        bound = np.count_nonzero(difference)*(p*q.norm/q.levels)**2
        assert np.dot(error, error) <= bound+1e-10


def test_combined_packet_rejects_invalid_norm_dtype_and_nested_payload():
    q = qsgd.quantize(np.array([1., -1.]), levels=3, rng=np.random.default_rng(1))
    packet = qsgd_llz.pack(q)
    for prefix in [struct.pack('<4sBd', b'QLP1', 1, float('nan')),
                   struct.pack('<4sBd', b'QLP1', 1, -1.),
                   struct.pack('<4sBd', b'QLP1', 0, 1.),
                   struct.pack('<4sBd', b'QLP1', 1, 0.)]:
        with pytest.raises(ValueError):
            qsgd_llz.decode(prefix+packet[qsgd_llz.HEADER.size:])
    for broken in [packet[:4], b'BAD1'+packet[4:], packet[:-1], packet+b'\x00']:
        with pytest.raises(ValueError):
            qsgd_llz.decode(broken)


def test_incompressible_input_is_reported_as_expansion_without_hidden_fallback():
    q = qsgd.QuantizedTensor(np.arange(128, dtype=np.int32), 1., 127, 'float32')
    packet = qsgd_llz.pack(q)
    assert len(packet) > len(qsgd.pack(q))
    np.testing.assert_array_equal(qsgd_llz.unpack(packet).codes, q.codes)


def test_maximum_window_bound_and_zero_length_field():
    x = np.array([-65535, 65535, 0], dtype=np.int32)
    packet = llz_p.encode(x, alphabet_bound=65535, window_size=65535)
    np.testing.assert_array_equal(llz_p.decode(packet), x)
    stats = llz_p.packet_stats(packet)
    assert stats['position_bits'] == 16 and stats['literal_bits'] == 17
    assert stats['length_bits'] == 0 and stats['triplet_count'] == 3
    assert stats['triplet_bits'] == 99 and stats['padding_bits'] == 5


def test_combined_encode_uses_same_rng_quantization_and_does_not_advance_it_twice():
    x = np.random.default_rng(3).normal(size=256).astype(np.float32)
    a, b = np.random.default_rng(17), np.random.default_rng(17)
    payload = qsgd_llz.encode(x, levels=31, rng=a)
    q = qsgd.quantize(x, levels=31, rng=b)
    assert payload == qsgd_llz.pack(q)
    assert a.random() == b.random()


def test_standalone_import_does_not_load_flower_or_torch():
    subprocess.run([sys.executable, '-c',
        "import sys; import compression.llz_p, compression.qsgd_llz; "
        "assert not any(k == 'torch' or k.startswith('torch.') or k == 'flwr' or k.startswith('flwr.') for k in sys.modules)"],
        check=True)
