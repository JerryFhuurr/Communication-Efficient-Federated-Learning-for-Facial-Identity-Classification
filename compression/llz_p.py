"""LLZ-p over signed integer symbols, with a self-contained LLP1 packet.

Semantics: supplied revised manuscript, Sec. III, Algorithms 2/3, PDF pp. 7-9.
See papers/README.md for source fingerprints, notation corrections, and framing.
No Flower, PyTorch, entropy coder, or cross-packet dictionary state.
"""

from dataclasses import dataclass
import math
import operator
import struct

import numpy as np


# magic, ndim, alphabet bound, p, window, maximum match length, triplet count
HEADER = struct.Struct('<4sBHIIII')
MAGIC = b'LLP1'
MAX_SYMBOLS = 10_000_000  # Default decoder allocation limit; caller can lower/raise.


@dataclass(frozen=True)
class SymbolPacket:
    symbols: np.ndarray
    alphabet_bound: int
    p: int
    window_size: int
    max_length: int
    triplet_count: int


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]') from error
    if not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def _settings(alphabet_bound, p, window_size):
    bound = _integer(alphabet_bound, 'alphabet_bound', 1, 65535)
    return bound, _integer(p, 'p', 0, 2*bound), _integer(window_size, 'window_size', 1, 65535)


def _symbols(symbols, bound):
    values = np.asarray(symbols)
    if values.dtype.kind != 'i' or values.ndim > 32:
        raise ValueError('symbols must be a signed integer array with at most 32 dimensions')
    if values.size > 0xFFFFFFFF or any(d > 0xFFFFFFFF for d in values.shape):
        raise ValueError('Symbol count and dimensions must fit uint32')
    if np.any(values < -bound) or np.any(values > bound):
        raise ValueError('Symbol outside declared alphabet')
    return values


def _parse(values, p, window):
    source = values.ravel(order='C').tolist()  # Python integers avoid subtraction overflow.
    reconstructed, triplets = [], []
    pointer, count = 0, len(source)
    while pointer < count:
        first = max(0, pointer-window)
        best_length, best_start = 0, 0
        # Descending starts implement the last/largest-position tie in Algorithm 2.
        for start in range(pointer-1, first-1, -1):
            limit = min(pointer-start, count-pointer-1)
            if limit <= best_length:
                continue
            length = 0
            while length < limit and abs(source[pointer+length]-reconstructed[start+length]) <= p:
                length += 1
            if length > best_length:
                best_length, best_start = length, start
        literal = source[pointer+best_length]
        position = best_start-first+1 if best_length else 0
        reconstructed.extend(reconstructed[best_start:best_start+best_length])
        reconstructed.append(literal)
        triplets.append((position, best_length, literal))
        pointer += best_length+1
    return triplets


def parse(symbols, *, alphabet_bound, p=0, window_size=128):
    """Return (1-based window position, match length, literal) triplets.

    Match only reconstructed history, forbid overlapping references, reserve a
    literal for every phrase, and choose the largest position for equal lengths.
    p is an integer number of quantization levels. Inputs are never modified.
    """
    bound, p, window = _settings(alphabet_bound, p, window_size)
    return _parse(_symbols(symbols, bound), p, window)


class _Writer:
    def __init__(self):
        self.data, self.buffer, self.used = bytearray(), 0, 0

    def write(self, value, width):
        self.buffer = (self.buffer << width) | value
        self.used += width
        while self.used >= 8:
            self.used -= 8
            self.data.append(self.buffer >> self.used)
            self.buffer &= (1 << self.used)-1

    def finish(self):
        if self.used:
            self.data.append(self.buffer << (8-self.used))
        return bytes(self.data)


class _Reader:
    def __init__(self, data):
        self.data, self.offset, self.buffer, self.used = data, 0, 0, 0

    def read(self, width):
        while self.used < width:
            self.buffer = (self.buffer << 8) | self.data[self.offset]
            self.offset += 1
            self.used += 8
        self.used -= width
        result = self.buffer >> self.used
        self.buffer &= (1 << self.used)-1
        return result


def encode(symbols, *, alphabet_bound, p=0, window_size=128):
    """Encode signed integer symbols into an LLP1 packet, including shape."""
    bound, p, window = _settings(alphabet_bound, p, window_size)
    values = _symbols(symbols, bound)
    triplets = _parse(values, p, window)
    maximum = max((t[1] for t in triplets), default=0)
    header = HEADER.pack(MAGIC, values.ndim, bound, p, window, maximum, len(triplets))
    shape = struct.pack('<'+'I'*values.ndim, *values.shape)
    writer = _Writer()
    for position, length, literal in triplets:
        writer.write(position, window.bit_length())
        writer.write(length, maximum.bit_length())
        writer.write(literal+bound, (2*bound).bit_length())
    return header+shape+writer.finish()


def unpack(payload, *, max_symbols=MAX_SYMBOLS):
    """Validate/decode a packet; reject invalid history references and framing."""
    if not isinstance(payload, bytes):
        raise TypeError('payload must be bytes')
    limit = _integer(max_symbols, 'max_symbols', 0, 0xFFFFFFFF)
    if len(payload) < HEADER.size:
        raise ValueError('Truncated LLP1 header')
    magic, ndim, bound, p, window, maximum, phrases = HEADER.unpack_from(payload)
    if magic != MAGIC or ndim > 32:
        raise ValueError('Unsupported LLP1 version or dimension count')
    _settings(bound, p, window)
    offset = HEADER.size+4*ndim
    if len(payload) < offset:
        raise ValueError('Truncated LLP1 shape')
    shape = struct.unpack_from('<'+'I'*ndim, payload, HEADER.size)
    count = math.prod(shape)
    if count > limit:
        raise ValueError('Decoded symbol count exceeds max_symbols')
    if maximum > min(window, max(0, count-1)) or not phrases <= count <= phrases*(maximum+1):
        raise ValueError('Invalid match length or triplet count')
    if count == 0 and (phrases or maximum):
        raise ValueError('Empty packet must have no triplets or matches')
    widths = (window.bit_length(), maximum.bit_length(), (2*bound).bit_length())
    bits = phrases*sum(widths)
    if len(payload) != offset+(bits+7)//8:
        raise ValueError('LLP1 payload length mismatch')
    padding = (-bits) % 8
    if padding and payload[-1] & ((1 << padding)-1):
        raise ValueError('Nonzero LLP1 padding')
    reader = _Reader(memoryview(payload)[offset:])
    result = np.empty(count, dtype=np.int32)
    pointer, observed_max = 0, 0
    for _ in range(phrases):
        position, length, symbol = (reader.read(width) for width in widths)
        if symbol > 2*bound or length > maximum or pointer+length+1 > count:
            raise ValueError('Invalid symbol, length, or decoded extent')
        first = max(0, pointer-window)
        start = first+position-1
        if length == 0:
            if position != 0:
                raise ValueError('Literal must have position zero')
        elif position == 0 or start < first or start+length > pointer:
            raise ValueError('Reference is outside reconstructed dictionary')
        else:
            result[pointer:pointer+length] = result[start:start+length]
        result[pointer+length] = symbol-bound
        pointer += length+1
        observed_max = max(observed_max, length)
    if pointer != count or observed_max != maximum:
        raise ValueError('Noncanonical maximum length or incomplete decoded sequence')
    return SymbolPacket(result.reshape(shape), bound, p, window, maximum, phrases)


def decode(payload, *, max_symbols=MAX_SYMBOLS):
    """Recover signed int32 symbols with their encoded shape."""
    return unpack(payload, max_symbols=max_symbols).symbols


def packet_stats(payload, *, max_symbols=MAX_SYMBOLS):
    """Count the paper's triplet bits separately from complete framing/padding."""
    packet = unpack(payload, max_symbols=max_symbols)
    widths = (packet.window_size.bit_length(), packet.max_length.bit_length(),
              (2*packet.alphabet_bound).bit_length())
    bits = packet.triplet_count*sum(widths)
    metadata = HEADER.size+4*packet.symbols.ndim
    return dict(codec='llz-p-fixed-v1', alphabet_bound=packet.alphabet_bound, p=packet.p,
                window_size=packet.window_size, shape=list(packet.symbols.shape),
                triplet_count=packet.triplet_count, max_length=packet.max_length,
                position_bits=widths[0], length_bits=widths[1], literal_bits=widths[2],
                triplet_bits=bits, metadata_bytes=metadata, padding_bits=(-bits) % 8,
                encoded_bytes=len(payload), encoded_bits=8*len(payload))
