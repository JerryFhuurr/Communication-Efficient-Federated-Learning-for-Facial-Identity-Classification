"""Standalone float32 block autoencoder, weight artifacts, and update packets.

This blockwise delta codec is an adaptation, not a paper reproduction. It has
no Flower dependency. AEW1/AEP1 fix architecture and block conventions at v1.
"""
import hashlib
import json
import struct

import numpy as np
import torch
from torch import nn

from compression.block_data import BLOCK_SIZE, normalize_blocks, padded_blocks

LATENT_SIZE = 6
CHUNK_SIZE = 1024
_PACKET = struct.Struct('<4sII32s')
_WEIGHTS = struct.Struct('<4sB')


class BlockAutoencoder(nn.Module):
    """256 -> 64 -> 6 -> 64 -> 256; linear latent and output layers."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(BLOCK_SIZE, 64), nn.ReLU(),
                                     nn.Linear(64, LATENT_SIZE))
        self.decoder = nn.Sequential(nn.Linear(LATENT_SIZE, 64), nn.ReLU(),
                                     nn.Linear(64, BLOCK_SIZE))

    def forward(self, blocks):
        return self.decoder(self.encoder(blocks))


def _validate_module(module, expected):
    actual = module.state_dict()
    reference = expected.state_dict()
    if type(module) is not type(expected) or list(actual) != list(reference):
        raise ValueError('Wrong codec architecture')
    # Check layers as well as weights: changing an activation changes decoding.
    if repr(module) != repr(expected):
        raise ValueError('Wrong codec architecture')
    for name, value in actual.items():
        if (value.shape != reference[name].shape or value.dtype != torch.float32
                or value.device.type != 'cpu' or not torch.isfinite(value).all()):
            raise ValueError('Codec weights must be finite CPU float32 with v1 shapes')


def _reference():
    # Validation/loading must not advance the caller's training RNG.
    with torch.random.fork_rng(devices=[]):
        return BlockAutoencoder()


def weight_bytes(model, *, decoder_only=False):
    """Serialize fixed-layout weights; decoder-only bytes are the setup payload.

    No pickle and no optimizer state. SHA256 of these exact decoder bytes is
    the decoder identity in every packet. A full artifact also stores encoder.
    """
    reference = _reference()
    target = model if not decoder_only else model.decoder
    _validate_module(target, reference if not decoder_only else reference.decoder)
    return _pack_weights(target, decoder_only)


def _pack_weights(target, decoder_only):
    body = _WEIGHTS.pack(b'AEW1', int(decoder_only)) + b''.join(
        v.detach().numpy().astype('<f4', copy=False).tobytes()
        for v in target.state_dict().values())
    return body + hashlib.sha256(body).digest()


def load_weights(payload, *, decoder_only=False):
    """Load strict v1 bytes, returning an eval-mode CPU model or decoder."""
    model = _reference()
    target = model.decoder if decoder_only else model
    if len(payload) < _WEIGHTS.size:
        raise ValueError('Truncated weight artifact')
    if _WEIGHTS.unpack_from(payload) != (b'AEW1', int(decoder_only)):
        raise ValueError('Wrong weight version or role')
    state = target.state_dict()
    expected = _WEIGHTS.size + sum(v.numel() * 4 for v in state.values()) + 32
    if len(payload) != expected:
        raise ValueError('Wrong weight artifact length')
    if hashlib.sha256(payload[:-32]).digest() != payload[-32:]:
        raise ValueError('Weight artifact checksum mismatch')
    offset = _WEIGHTS.size
    for name, value in state.items():
        size = value.numel() * 4
        a = np.frombuffer(payload, dtype='<f4', count=value.numel(), offset=offset)
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite weights')
        state[name] = torch.from_numpy(a.copy().reshape(tuple(value.shape)))
        offset += size
    target.load_state_dict(state)
    return target.eval()


def _layout(shapes):
    if not shapes or not all(isinstance(k, str) and k for k in shapes):
        raise ValueError('Nonempty named tensor layout required')
    result = []
    for name in sorted(shapes):
        shape = tuple(shapes[name])
        if any(type(n) is not int or n < 1 for n in shape):
            raise ValueError('Positive integer dimensions required')
        size = 1
        for n in shape:
            size *= n
        result.append((name, shape, size, (size + BLOCK_SIZE - 1) // BLOCK_SIZE))
    header = json.dumps([[name, list(shape)] for name, shape, _, _ in result],
                        ensure_ascii=True, separators=(',', ':')).encode('ascii')
    if len(header) > 1024 * 1024 or sum(x[3] for x in result) > 2**32 - 1:
        raise ValueError('Layout exceeds v1 limits')
    return result, header


def _infer(module, values, output_size):
    if module.training:
        raise ValueError('Codec inference requires eval mode')
    result = np.empty((len(values), output_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(values), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(values))
            result[start:end] = module(torch.from_numpy(values[start:end])).numpy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite codec inference')
    return result


def encode(tensors, model):
    """Encode sorted, tensor-local blocks; return actual packet bytes."""
    weights = weight_bytes(model)
    if model.training or model.encoder.training or model.decoder.training:
        raise ValueError('Codec inference requires eval mode')
    layout, header = _layout({k: tuple(v.shape) for k, v in tensors.items()})
    # Full validation above includes encoder and decoder; no model mutation.
    del weights
    digest = hashlib.sha256(weight_bytes(model, decoder_only=True)).digest()
    chunks = []
    for name, _, _, _ in layout:
        blocks, scales = normalize_blocks(padded_blocks(tensors[name]))
        latent = _infer(model.encoder, blocks, LATENT_SIZE)
        latent[scales == 0] = 0
        chunks.append(np.column_stack((scales, latent)).astype('<f4').tobytes())
    count = sum(x[3] for x in layout)
    return _PACKET.pack(b'AEP1', len(header), count, digest) + header + b''.join(chunks)


def decode(packet, decoder, expected_shapes):
    """Decode only against trusted expected shapes and a matching decoder.

    Cropping discards each tensor's own padding. Zero scales bypass decoder
    biases and reconstruct exactly zero. No shape is allocated from packet JSON.
    """
    reference = _reference().decoder
    _validate_module(decoder, reference)
    if decoder.training:
        raise ValueError('Codec inference requires eval mode')
    layout, header = _layout(expected_shapes)
    if len(packet) < _PACKET.size:
        raise ValueError('Truncated packet')
    magic, length, count, digest = _PACKET.unpack_from(packet)
    expected_count = sum(x[3] for x in layout)
    if magic != b'AEP1' or length != len(header) or count != expected_count:
        raise ValueError('Wrong packet version or layout')
    expected_length = _PACKET.size + len(header) + expected_count * 28
    if len(packet) != expected_length:
        raise ValueError('Wrong packet length')
    if packet[_PACKET.size:_PACKET.size + length] != header:
        raise ValueError('Tensor layout mismatch')
    decoder_bytes = _pack_weights(decoder, True)
    if digest != hashlib.sha256(decoder_bytes).digest():
        raise ValueError('Decoder identity mismatch')
    records = np.frombuffer(packet, dtype='<f4', offset=_PACKET.size + length).reshape(count, 7)
    if not np.isfinite(records).all() or np.any(records[:, 0] < 0):
        raise ValueError('Invalid scales or latents')
    if np.any(records[records[:, 0] == 0, 1:] != 0):
        raise ValueError('Noncanonical zero block')
    result = {}
    offset = 0
    for name, shape, size, n in layout:
        rows = records[offset:offset + n]
        reconstructed = np.zeros((n, BLOCK_SIZE), np.float32)
        nonzero = rows[:, 0] != 0
        with np.errstate(over='ignore', invalid='ignore'):
            reconstructed[nonzero] = _infer(decoder, rows[nonzero, 1:].copy(), BLOCK_SIZE) * rows[nonzero, :1]
        if not np.isfinite(reconstructed).all():
            raise ValueError('Nonfinite reconstructed update')
        result[name] = reconstructed.reshape(-1)[:size].reshape(shape).copy()
        offset += n
    return result
