"""Logical per-recipient communication accounting for Flower 1.36 messages.

This measures serialized object bytes, not network traffic. Each message includes
its full object graph with identical objects counted once within that message.
Objects are counted again for each recipient and each subsequent message.
"""

from collections import Counter
import json
import math

import numpy as np


MEASUREMENT = {
    "schema_version": 4,
    "boundary": "logical per-recipient Flower Message object graph at the server Grid boundary",
    "serializer": "Flower 1.36 deflate() on Message and all unique descendant objects",
    "observation": "outgoing messages after Grid dispatch returns or raises; incoming replies before aggregation",
    "deduplication": "within each message by object ID; none across messages, recipients, phases, or rounds",
    "downlink": "server-to-client instructions submitted to Grid; delivery is not inferred",
    "uplink": "client-to-server replies returned by Grid, including error replies",
    "raw_array_bytes": "sum of shape elements times dtype itemsize for every named array (including duplicates)",
    "array_data_bytes": "sum of len(Array.data) for every named array, including NumPy headers and duplicates",
    "codec_payload_bytes": "sum of QSD1 or QLP1 byte packets in the qsgd ConfigRecord, including each packet header and padding",
    "serialized_object_bytes": "sum of len(object.deflate()) for each unique object in one message",
    "bits": "8 times serialized_object_bytes",
    "includes": ["array data and headers", "array names, shapes, and dtypes", "configuration and metrics",
                 "message metadata as observed", "object headers and child references"],
    "excludes": ["transport/RPC envelopes and separate object-tree announcements", "HTTP/TLS/TCP/IP headers",
                 "polling, acknowledgements, retries, and runtime control traffic", "runtime cache savings",
                 "app bundle, dataset access, checkpoints, and local test evaluation"],
    "network_traffic_measured": False,
}


def measure_message(message):
    """Serialize the actual object graph; never estimate from parameter count."""
    objects = {}
    pending = [message]
    by_type = {}
    while pending:
        obj = pending.pop()
        object_id = obj.object_id
        if object_id in objects:
            continue
        size = len(obj.deflate())
        objects[object_id] = size
        kind = type(obj).__name__
        entry = by_type.setdefault(kind, {"objects": 0, "bytes": 0})
        entry["objects"] += 1
        entry["bytes"] += size
        pending.extend((obj.children or {}).values())

    raw_bytes = data_bytes = codec_bytes = 0
    if message.has_content():
        for record in message.content.array_records.values():
            for array in record.values():
                raw_bytes += math.prod(array.shape) * np.dtype(array.dtype).itemsize
                data_bytes += len(array.data)
        if "qsgd" in message.content.config_records:
            codec_bytes = sum(len(packet) for packet in message.content["qsgd"].values() if isinstance(packet, bytes))
    serialized_bytes = sum(objects.values())
    return dict(raw_array_bytes=raw_bytes, array_data_bytes=data_bytes, codec_payload_bytes=codec_bytes,
                serialized_object_bytes=serialized_bytes, serialized_object_bits=8 * serialized_bytes,
                objects_by_type=by_type)


COUNTERS = ("messages", "error_messages", "raw_array_bytes", "array_data_bytes",
            "codec_payload_bytes", "serialized_object_bytes", "serialized_object_bits")


class CommunicationLedger:
    """Save message observations immediately, including incomplete exchanges."""

    def __init__(self, experiment):
        self.experiment = experiment
        self.rounds = {}
        experiment.metadata["communication_measurement"] = MEASUREMENT
        experiment.write_json("experiment.json", experiment.metadata)
        self.save()

    def observe(self, messages, server_round, phase, direction):
        if phase not in ("train", "validation") or direction not in ("downlink", "uplink"):
            raise ValueError("Unknown communication phase or direction")
        buckets = self.rounds.setdefault(server_round, {})
        bucket = buckets.setdefault((phase, direction), Counter())
        with (self.experiment.output / "communication_messages.jsonl").open("a", encoding="utf-8") as stream:
            for message in messages:
                sizes = measure_message(message)
                row = dict(round=server_round, phase=phase, direction=direction,
                           client_node_id=(message.metadata.dst_node_id if direction == "downlink"
                                           else message.metadata.src_node_id),
                           message_id=message.metadata.message_id,
                           reply_to_message_id=message.metadata.reply_to_message_id,
                           run_id=message.metadata.run_id, has_error=message.has_error(), **sizes)
                client = message.content.config_records.get("client") if message.has_content() else None
                row["client_partition_id"] = client.get("partition-id") if client is not None else None
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                bucket.update({"messages": 1, "error_messages": int(message.has_error()),
                               **{key: sizes[key] for key in COUNTERS if key in sizes}})

    def summary(self, server_round=None):
        combined = {}
        selected = self.rounds.values() if server_round is None else (self.rounds.get(server_round, {}),)
        for buckets in selected:
            for key, value in buckets.items():
                combined.setdefault(key, Counter()).update(value)

        def total(phase=None, direction=None):
            count = Counter()
            for (p, d), value in combined.items():
                if (phase is None or phase == p) and (direction is None or direction == d):
                    count.update(value)
            return {key: count[key] for key in COUNTERS}

        return dict(total=total(), downlink=total(direction="downlink"), uplink=total(direction="uplink"),
                    **{phase: {"total": total(phase), "downlink": total(phase, "downlink"),
                               "uplink": total(phase, "uplink")} for phase in ("train", "validation")})

    def save(self):
        self.experiment.write_json("communication.json", dict(
            measurement=MEASUREMENT, totals=self.summary(),
            rounds={str(step): self.summary(step) for step in sorted(self.rounds)},
        ))


class MeasuredGrid:
    """Delegate FedAvg's Grid calls without adding messages or changing payloads."""

    def __init__(self, grid, ledger):
        self.grid = grid
        self.ledger = ledger

    def __getattr__(self, name):
        return getattr(self.grid, name)

    def send_and_receive(self, messages, *, timeout=None):
        messages = list(messages)
        if not messages:
            return self.grid.send_and_receive(messages, timeout=timeout)
        batches = {(msg.metadata.message_type, msg.content["config"]["server-round"])
                   for msg in messages}
        if len(batches) != 1:
            raise ValueError("Expected one FedAvg phase and round per Grid exchange")
        message_type, server_round = batches.pop()
        phase = {"train": "train", "evaluate": "validation"}[message_type]
        replies = []
        try:
            # Grid fills run/source/message IDs during dispatch. Observe afterwards
            # so these fields are included, before FedAvg can modify any arrays.
            for reply in self.grid.send_and_receive(messages, timeout=timeout):
                replies.append(reply)
        finally:
            self.ledger.observe(messages, server_round, phase, "downlink")
            self.ledger.observe(replies, server_round, phase, "uplink")
            self.ledger.save()
        return replies
