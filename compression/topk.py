"""Deterministic global Top-k packets; NumPy only, float32 values/uint32 indices."""
import struct
import numpy as np

HEADER = struct.Struct('<4sII')


def encode(values, k):
    x = np.asarray(values)
    if x.dtype != np.float32 or x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Expected a finite float32 vector')
    n = x.size
    if not 0 < n < 2**32 or type(k) is not int or not 1 <= k <= n:
        raise ValueError('Require 1 <= k <= vector length < 2**32')
    magnitude = np.abs(x)
    threshold = np.partition(magnitude, n-k)[n-k]
    above = np.flatnonzero(magnitude > threshold)
    ties = np.flatnonzero(magnitude == threshold)[:k-above.size]
    indices = np.sort(np.concatenate((above, ties))).astype('<u4')
    return HEADER.pack(b'TPK1', n, k) + indices.tobytes() + x[indices].astype('<f4').tobytes()


def decode(packet, *, expected_size, expected_k):
    if not isinstance(packet, bytes) or len(packet) < HEADER.size:
        raise ValueError('Truncated Top-k packet')
    magic, n, k = HEADER.unpack_from(packet)
    if magic != b'TPK1' or n != expected_size or k != expected_k or not 1 <= k <= n:
        raise ValueError('Top-k header mismatch')
    if len(packet) != HEADER.size + 8*k:
        raise ValueError('Top-k packet length mismatch')
    indices = np.frombuffer(packet, dtype='<u4', count=k, offset=HEADER.size)
    values = np.frombuffer(packet, dtype='<f4', count=k, offset=HEADER.size+4*k)
    if indices[-1] >= n or np.any(indices[1:] <= indices[:-1]) or not np.isfinite(values).all():
        raise ValueError('Invalid Top-k indices or values')
    result = np.zeros(n, dtype=np.float32)
    result[indices] = values
    return result
