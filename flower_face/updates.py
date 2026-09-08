"""PyTorch/Flower adapter for standalone QSGD model-delta packets."""

import math

from flwr.app import ConfigRecord, MetricRecord
import numpy as np
import torch

from compression.qsgd import dequantize, encode, unpack

CODEC = "qsgd-l2-dense-v1"


def compression_settings(config):
    method = config.get("compression", "none")
    levels = config.get("qsgd-levels", 127)
    if method not in ("none", "qsgd"):
        raise ValueError("compression must be 'none' or 'qsgd'")
    if isinstance(levels, bool) or not isinstance(levels, int) or not 1 <= levels <= 65535:
        raise ValueError("qsgd-levels must be an integer in [1, 65535]")
    return method, levels


def encode_update(trained, reference, *, levels, seed, server_round, client_id):
    """Use a separate codec RNG and one QSD1 packet per named model tensor."""
    if set(trained) != set(reference):
        raise ValueError("Model tensor names differ from reference")
    rng = np.random.default_rng(np.random.SeedSequence([seed, server_round, client_id, 0x51534744]))
    packets = {}
    for name in sorted(reference):
        value, base = trained[name].detach().cpu(), reference[name].detach().cpu()
        if value.shape != base.shape or value.dtype != base.dtype:
            raise ValueError(f"Model shape/dtype mismatch: {name}")
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"QSGD requires floating model tensors: {name}")
        packets[name] = encode((value - base).numpy(), levels=levels, rng=rng)
    return ConfigRecord(packets)


def decode_update(content, reference, *, levels, server_round):
    """Validate the complete update before it enters weighted aggregation."""
    if set(content) != {"qsgd", "update", "metrics"}:
        raise ValueError("Expected QSGD packets, update metadata, and metrics only")
    packets, metadata, metrics = content["qsgd"], content["update"], content["metrics"]
    if not isinstance(packets, ConfigRecord) or not isinstance(metadata, ConfigRecord):
        raise ValueError("QSGD payload and metadata must be ConfigRecords")
    if dict(metadata) != {"codec": CODEC, "server-round": server_round, "levels": levels}:
        raise ValueError("QSGD codec, round, or level metadata mismatch")
    if not isinstance(metrics, MetricRecord):
        raise ValueError("Expected client metrics")
    count = metrics.get("num-examples")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Client sample count must be a positive integer")
    loss = metrics.get("train_loss")
    if not isinstance(loss, (float, int)) or not math.isfinite(loss):
        raise ValueError("Client training loss must be finite")
    if set(packets) != set(reference):
        raise ValueError("QSGD tensor names differ from reference")
    decoded = {}
    for name, base in reference.items():
        q = unpack(packets[name])
        if q.levels != levels or q.codes.shape != tuple(base.shape) or q.dtype != str(base.numpy().dtype):
            raise ValueError(f"QSGD packet levels/shape/dtype mismatch: {name}")
        decoded[name] = torch.from_numpy(dequantize(q))
    return decoded
