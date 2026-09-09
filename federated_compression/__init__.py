"""Framework adapters for compressed federated model updates.

The codecs themselves remain in :mod:`compression` and have no Flower,
PyTorch, model, or dataset dependency.
"""

from federated_compression.updates import (
    CODEC,
    CODECS,
    compression_settings,
    decode_update,
    encode_update,
    llz_settings,
    update_metadata,
)

__all__ = [
    "CODEC",
    "CODECS",
    "compression_settings",
    "decode_update",
    "encode_update",
    "llz_settings",
    "update_metadata",
]
