"""L2 QSGD quantization with a project-defined dense binary encoding.

Quantizer: Alistarh et al., NeurIPS 2017, Section 3.1, Eq. (4).
https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf
The encoding below is fixed-width, not the paper's Elias encoding.
"""

from dataclasses import dataclass
import math
import operator
import struct

import numpy as np


# Magic/version, dtype code, dimension count, levels, float64 L2 norm.
HEADER = struct.Struct("<4sBBHd")
MAGIC = b"QSD1"
DTYPES = {1: np.dtype("float32"), 2: np.dtype("float64")}
MAX_NDIM = 32


@dataclass(frozen=True)
class QuantizedTensor:
    """Signed integer levels plus reconstruction metadata, before byte encoding."""

    codes: np.ndarray
    norm: float
    levels: int
    dtype: str


def _levels(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("levels must be an integer in [1, 65535]")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError("levels must be an integer in [1, 65535]") from error
    if not 1 <= value <= 65535:
        raise ValueError("levels must be an integer in [1, 65535]")
    return value


def _validate(q):
    levels = _levels(q.levels)
    if q.dtype not in ("float32", "float64"):
        raise ValueError("Only float32 and float64 are supported")
    if not isinstance(q.codes, np.ndarray) or q.codes.dtype.kind != "i":
        raise ValueError("codes must be a signed integer NumPy array")
    if q.codes.ndim > MAX_NDIM or any(d > 0xFFFFFFFF for d in q.codes.shape):
        raise ValueError("Unsupported tensor shape")
    if not math.isfinite(q.norm) or q.norm < 0:
        raise ValueError("norm must be finite and nonnegative")
    if np.any(q.codes < -levels) or np.any(q.codes > levels):
        raise ValueError("Integer code outside quantization range")
    if q.norm == 0 and np.any(q.codes != 0):
        raise ValueError("Zero norm requires zero codes")
    if q.codes.size == 0 and q.norm != 0:
        raise ValueError("Empty tensors require zero norm")
    peak = max(int(np.max(q.codes, initial=0)), -int(np.min(q.codes, initial=0)))
    if (peak / levels) * q.norm > float(np.finfo(q.dtype).max):
        raise ValueError("Reconstruction exceeds output dtype range")
    return levels


def quantize(array, *, levels, rng):
    """Stochastically round each coordinate using one L2 norm for this tensor.

    ``levels`` is s (positive intervals), not a bit count. Pass an explicit
    NumPy Generator; each call advances it without using global RNG state.
    Computation uses float64; reconstruction returns the original float dtype.
    """
    levels = _levels(levels)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    array = np.asarray(array)
    if array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8):
        raise ValueError("Input must be a float32 or float64 array")
    if array.ndim > MAX_NDIM or any(d > 0xFFFFFFFF for d in array.shape):
        raise ValueError("Unsupported tensor shape")
    values = array.astype(np.float64).ravel(order="C")
    if not np.isfinite(values).all():
        raise ValueError("Input must contain only finite values")
    # Scale first so squaring large/small finite coordinates does not overflow
    # or underflow unnecessarily. Reject norms outside float64's finite range.
    largest = float(np.max(np.abs(values), initial=0.0))
    norm = largest * float(np.linalg.norm(values / largest)) if largest else 0.0
    if not math.isfinite(norm):
        raise ValueError("L2 norm exceeds float64 range")
    codes = np.zeros(values.size, dtype=np.int32)
    if norm:
        scaled = np.clip((np.abs(values) / norm) * levels, 0, levels)
        lower = np.floor(scaled)
        magnitude = (lower + (rng.random(values.size) < scaled - lower)).astype(np.int32)
        codes = np.where(values < 0, -magnitude, magnitude)
    return QuantizedTensor(codes.reshape(array.shape), norm, levels,
                           "float32" if array.dtype.itemsize == 4 else "float64")


def dequantize(q):
    """Reconstruct numeric values from signed levels (lossy, except special cases)."""
    _validate(q)
    values = (q.codes.astype(np.float64) / q.levels) * q.norm
    with np.errstate(over="ignore"):
        result = values.astype(q.dtype)
    if not np.isfinite(result).all():
        raise ValueError("Reconstruction exceeds output dtype range")
    return result


def pack(q):
    """Losslessly serialize quantized codes and all reconstruction metadata."""
    levels = _validate(q)
    dtype_code = 1 if q.dtype == "float32" else 2
    header = HEADER.pack(MAGIC, dtype_code, q.codes.ndim, levels, q.norm)
    shape = struct.pack("<" + "I" * q.codes.ndim, *q.codes.shape)
    width = levels.bit_length()  # ceil(log2(s + 1)), including the endpoint s
    flat = q.codes.ravel(order="C").astype(np.int32)
    symbols = np.abs(flat).astype(np.uint32) | ((flat < 0).astype(np.uint32) << width)
    # One sign bit followed by magnitude bits, MSB first; trailing padding is 0.
    bits = ((symbols[:, None] >> np.arange(width, -1, -1, dtype=np.uint32)) & 1).astype(np.uint8)
    return header + shape + np.packbits(bits.ravel(), bitorder="big").tobytes()


def unpack(payload):
    """Decode a self-contained QSD1 packet, rejecting malformed representations."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if len(payload) < HEADER.size:
        raise ValueError("Truncated QSD1 header")
    magic, dtype_code, ndim, levels, norm = HEADER.unpack_from(payload)
    if magic != MAGIC or dtype_code not in DTYPES or ndim > MAX_NDIM:
        raise ValueError("Unsupported QSD1 version, dtype, or dimension count")
    _levels(levels)
    if not math.isfinite(norm) or norm < 0:
        raise ValueError("norm must be finite and nonnegative")
    offset = HEADER.size + 4 * ndim
    if len(payload) < offset:
        raise ValueError("Truncated shape")
    shape = struct.unpack_from("<" + "I" * ndim, payload, HEADER.size)
    count = math.prod(shape)
    width = levels.bit_length()
    code_bits = count * (width + 1)
    if len(payload) != offset + (code_bits + 7) // 8:
        raise ValueError("Payload length does not match shape and levels")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8, offset=offset), bitorder="big")
    if np.any(bits[code_bits:]):
        raise ValueError("Nonzero padding bits")
    words = bits[:code_bits].reshape(count, width + 1).astype(np.uint32)
    symbols = np.sum(words << np.arange(width, -1, -1, dtype=np.uint32), axis=1, dtype=np.uint32)
    magnitude = (symbols & ((1 << width) - 1)).astype(np.int32)
    negative = (symbols >> width).astype(bool)
    if np.any(negative & (magnitude == 0)):
        raise ValueError("Noncanonical negative zero code")
    codes = np.where(negative, -magnitude, magnitude).reshape(shape)
    q = QuantizedTensor(codes, norm, levels, DTYPES[dtype_code].name)
    _validate(q)
    return q


def encode(array, *, levels, rng):
    """Quantize and serialize to bytes; the decoder needs no external metadata."""
    return pack(quantize(array, levels=levels, rng=rng))


def decode(payload):
    """Deserialize and reconstruct an array with its original shape and dtype."""
    return dequantize(unpack(payload))


def packet_stats(payload):
    """Count real packet bytes, including metadata and byte-alignment padding."""
    q = unpack(payload)
    metadata_bytes = HEADER.size + 4 * q.codes.ndim
    code_bits = q.codes.size * (q.levels.bit_length() + 1)
    return dict(codec="qsgd-l2-dense-v1", levels=q.levels, shape=list(q.codes.shape), dtype=q.dtype,
                raw_bytes=q.codes.size * np.dtype(q.dtype).itemsize,
                metadata_bytes=metadata_bytes, code_bits=code_bits,
                padding_bits=8 * (len(payload) - metadata_bytes) - code_bits,
                encoded_bytes=len(payload), encoded_bits=8 * len(payload))
