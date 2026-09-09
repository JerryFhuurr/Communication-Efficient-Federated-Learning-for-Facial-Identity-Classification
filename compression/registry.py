"""Packet backends for QSGD integer codes, independent of the FL framework.

New QSGD symbol encoders implement pack/unpack/stats. Algorithms with different
quantizers also require an update policy; this registry does not pretend that
Top-k is a QSGD symbol encoder.
"""
from dataclasses import dataclass
from typing import Callable
from compression import qsgd, qsgd_llz


@dataclass(frozen=True)
class PacketCodec:
    wire_id: str
    pack: Callable
    unpack: Callable
    stats: Callable
    secondary: bool = False


REGISTRY = {
    'qsgd': PacketCodec('qsgd-l2-dense-v1',
        lambda q, p, window: qsgd.pack(q),
        lambda packet, count, p, window: qsgd.unpack(packet), qsgd.packet_stats),
    'qsgd-llz': PacketCodec('qsgd-llz-p-v1',
        lambda q, p, window: qsgd_llz.pack(q, p=p, window_size=window),
        lambda packet, count, p, window: qsgd_llz.unpack(packet,
            max_symbols=count, expected_p=p, expected_window_size=window),
        qsgd_llz.packet_stats, True),
}


def get_codec(name):
    try:
        return REGISTRY[name]
    except KeyError as error:
        raise ValueError(f'Unknown packet codec: {name}') from error
