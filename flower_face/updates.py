"""PyTorch/Flower adapter for standalone QSGD model-delta packets."""

import math

from flwr.app import ConfigRecord, MetricRecord
import numpy as np
import torch

from compression import qsgd, qsgd_llz

CODEC = "qsgd-l2-dense-v1"
CODECS = {"qsgd": CODEC, "qsgd-llz": "qsgd-llz-p-v1"}


def compression_settings(config):
    method = config.get("compression", "none")
    levels = config.get("qsgd-levels", 127)
    if method not in ("none", "qsgd", "qsgd-llz"):
        raise ValueError("compression must be 'none', 'qsgd', or 'qsgd-llz'")
    if isinstance(levels, bool) or not isinstance(levels, int) or not 1 <= levels <= 65535:
        raise ValueError("qsgd-levels must be an integer in [1, 65535]")
    if method == "qsgd-llz":
        llz_settings(config)
    return method, levels


def llz_settings(config):
    p, window = config.get("llz-p", 0), config.get("llz-window", 128)
    if type(p) is not int or p != 0:
        raise ValueError("Flower integration currently requires llz-p=0")
    if type(window) is not int or not 1 <= window <= 65535:
        raise ValueError("llz-window must be an integer in [1, 65535]")
    return p, window


def update_metadata(*, method="qsgd", levels, server_round, llz_p=0, llz_window=128):
    if method not in CODECS:
        raise ValueError("Expected a supported QSGD upload method")
    metadata = {"codec": CODECS[method], "server-round": server_round, "levels": levels}
    if method == "qsgd-llz":
        llz_settings({"llz-p": llz_p, "llz-window": llz_window})
        metadata.update({"llz-p": llz_p, "llz-window": llz_window})
    return metadata


def encode_update(trained, reference, *, levels, seed, server_round, client_id,
                  method="qsgd", llz_p=0, llz_window=128):
    """Use identical QSGD draws for either packet format, one packet per tensor."""
    update_metadata(method=method, levels=levels, server_round=server_round, llz_p=llz_p, llz_window=llz_window)
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
        quantized = qsgd.quantize((value - base).numpy(), levels=levels, rng=rng)
        packets[name] = (qsgd_llz.pack(quantized, p=llz_p, window_size=llz_window)
                         if method == "qsgd-llz" else qsgd.pack(quantized))
    return ConfigRecord(packets)


def decode_update(content, reference, *, levels, server_round, method="qsgd", llz_p=0, llz_window=128):
    """Validate the complete update before it enters weighted aggregation."""
    if set(content) != {"qsgd", "update", "metrics", "client"}:
        raise ValueError("Expected QSGD packets, update metadata, and metrics only")
    packets, metadata, metrics = content["qsgd"], content["update"], content["metrics"]
    if not isinstance(packets, ConfigRecord) or not isinstance(metadata, ConfigRecord):
        raise ValueError("QSGD payload and metadata must be ConfigRecords")
    expected = update_metadata(method=method, levels=levels, server_round=server_round,
                               llz_p=llz_p, llz_window=llz_window)
    if (dict(metadata) != expected
            or any(type(metadata[key]) is not type(value) for key, value in expected.items())):
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
        q = (qsgd_llz.unpack(packets[name], max_symbols=base.numel(), expected_p=llz_p,
                             expected_window_size=llz_window)
             if method == "qsgd-llz" else qsgd.unpack(packets[name]))
        if q.levels != levels or q.codes.shape != tuple(base.shape) or q.dtype != str(base.numpy().dtype):
            raise ValueError(f"QSGD packet levels/shape/dtype mismatch: {name}")
        decoded[name] = torch.from_numpy(qsgd.dequantize(q))
    return decoded
