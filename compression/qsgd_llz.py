"""Standalone QSGD + LLZ-p packets. No Flower or PyTorch integration."""

import struct

from compression import llz_p, qsgd


HEADER = struct.Struct('<4sBd')  # magic/version, float dtype code, exact float64 norm
MAGIC = b'QLP1'


def pack(q, *, p=0, window_size=128):
    """Encode an already quantized tensor, without drawing any new randomness."""
    qsgd._validate(q)
    symbols = llz_p.encode(q.codes, alphabet_bound=q.levels, p=p, window_size=window_size)
    return HEADER.pack(MAGIC, 1 if q.dtype == 'float32' else 2, q.norm)+symbols


def unpack(payload, *, max_symbols=llz_p.MAX_SYMBOLS, expected_p=None, expected_window_size=None):
    if not isinstance(payload, bytes):
        raise TypeError('payload must be bytes')
    if len(payload) < HEADER.size:
        raise ValueError('Truncated QLP1 header')
    magic, dtype, norm = HEADER.unpack_from(payload)
    if magic != MAGIC or dtype not in qsgd.DTYPES:
        raise ValueError('Unsupported QLP1 version or dtype')
    symbols = llz_p.unpack(payload[HEADER.size:], max_symbols=max_symbols)
    if ((expected_p is not None and symbols.p != expected_p)
            or (expected_window_size is not None and symbols.window_size != expected_window_size)):
        raise ValueError('LLZ packet distortion/window settings mismatch')
    q = qsgd.QuantizedTensor(symbols.symbols, norm, symbols.alphabet_bound, qsgd.DTYPES[dtype].name)
    qsgd._validate(q)
    return q


def encode(array, *, levels, rng, p=0, window_size=128):
    return pack(qsgd.quantize(array, levels=levels, rng=rng), p=p, window_size=window_size)


def decode(payload, *, max_symbols=llz_p.MAX_SYMBOLS):
    return qsgd.dequantize(unpack(payload, max_symbols=max_symbols))


def packet_stats(payload, *, max_symbols=llz_p.MAX_SYMBOLS):
    q = unpack(payload, max_symbols=max_symbols)
    result = llz_p.packet_stats(payload[HEADER.size:], max_symbols=max_symbols)
    result.update(codec='qsgd-llz-p-v1', levels=q.levels, dtype=q.dtype,
                  raw_bytes=q.codes.size*qsgd.DTYPES[1 if q.dtype == 'float32' else 2].itemsize,
                  metadata_bytes=HEADER.size+result['metadata_bytes'],
                  encoded_bytes=len(payload), encoded_bits=8*len(payload))
    return result
