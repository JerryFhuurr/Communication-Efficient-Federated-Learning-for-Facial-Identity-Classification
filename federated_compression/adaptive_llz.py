"""Flower framing and validation for the independent adaptive LLZ selector."""

import math
from time import perf_counter

from flwr.app import ConfigRecord, Message, MetricRecord, RecordDict
import numpy as np
import torch

from compression import adaptive_llz
from federated_compression.updates import decode_update, llz_settings, update_metadata

METHOD = 'qsgd-llz-adaptive'
WIRE_ID = 'qsgd-llz-adaptive-v1'


def settings(config, *, levels=127):
    budget = adaptive_llz.validate_budget(config.get('llz-budget', 0.0))
    _, window = llz_settings({'llz-p': 0, 'llz-window': config.get('llz-window', 128)}, levels=levels)
    return budget, window


def metadata(*, levels, server_round, p, window, budget):
    return dict(codec=WIRE_ID, **{'server-round': server_round, 'levels': levels,
        'llz-p': p, 'llz-window': window, 'llz-budget': float(budget),
        'distortion': 'additional-llz-over-original-squared-norm-v1'})


def encode_reply(trained, reference, *, levels, seed, server_round, client_id,
                 budget, window, metrics, instruction, measure):
    """Select by full reply bytes, including diagnostics, before sending once."""
    started = perf_counter()
    if set(trained) != set(reference):
        raise ValueError('Model tensor names differ from reference')
    updates = {}
    for name in sorted(reference):
        value, base = trained[name].detach().cpu(), reference[name].detach().cpu()
        if value.shape != base.shape or value.dtype != base.dtype:
            raise ValueError(f'Model shape/dtype mismatch: {name}')
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError(f'QSGD requires floating model tensors: {name}')
        updates[name] = (value - base).numpy()

    def message(candidate, candidates, costs, seconds):
        # Fixed-width double diagnostics make the byte-cost preflight identical
        # in size to the final record even before costs/timing are filled in.
        diagnostics = dict(candidate.stats, adaptive_selected_p=candidate.p,
            adaptive_budget=float(budget), adaptive_selection_seconds=float(seconds),
            adaptive_candidate_p=[float(c.p) for c in candidates],
            adaptive_candidate_payload_bytes=[float(c.payload_bytes) for c in candidates],
            adaptive_candidate_message_bytes=list(map(float, costs)),
            adaptive_candidate_squared_error=[c.stats['llz_reconstruction_squared_error'] for c in candidates],
            adaptive_candidate_total_squared_error=[c.stats['llz_total_squared_error'] for c in candidates],
            adaptive_selected_message_bytes=float(costs[candidate.p]))
        content = RecordDict({'qsgd': ConfigRecord(candidate.packets),
            'update': ConfigRecord(metadata(levels=levels, server_round=server_round,
                p=candidate.p, window=window, budget=budget)),
            'metrics': MetricRecord(dict(metrics, **diagnostics)),
            'client': ConfigRecord({'partition-id': client_id})})
        return Message(content=content, reply_to=instruction)

    def cost(candidate, candidates):
        return measure(message(candidate, candidates, (0, 0, 0), 0.0))['serialized_object_bytes']

    selection = adaptive_llz.select(updates, levels=levels,
        rng=np.random.default_rng(np.random.SeedSequence([seed, server_round, client_id, 0x51534744])),
        budget=budget, window_size=window, cost=cost)
    reply = message(selection.selected, selection.candidates, selection.costs, perf_counter()-started)
    if measure(reply)['serialized_object_bytes'] != selection.costs[selection.selected.p]:
        raise RuntimeError('Adaptive reply framing changed size after selection')
    return reply


def decode(content, reference, *, levels, server_round, budget, window):
    """Validate framing and reported budget, then decode the selected packet.

    The server cannot independently verify error against an unavailable client
    update. It checks internal consistency; honest client measurements remain
    an explicit assumption, as with the fixed-p diagnostics.
    """
    budget = adaptive_llz.validate_budget(budget)
    if set(content) != {'qsgd', 'update', 'metrics', 'client'}:
        raise ValueError('Expected adaptive packets, metadata, metrics and client')
    info, metrics = content['update'], content['metrics']
    if not isinstance(info, ConfigRecord) or not isinstance(metrics, MetricRecord):
        raise ValueError('Invalid adaptive records')
    p = info.get('llz-p')
    if type(p) is not int or p not in adaptive_llz.CANDIDATES:
        raise ValueError('Invalid adaptive selected p')
    expected = metadata(levels=levels, server_round=server_round, p=p, window=window, budget=budget)
    if (dict(info) != expected or any(type(info[k]) is not type(v) for k,v in expected.items())):
        raise ValueError('Adaptive codec, round or policy metadata mismatch')
    scalar_keys = ('adaptive_budget', 'adaptive_selection_seconds', 'adaptive_selected_message_bytes',
        'qsgd_input_squared_norm', 'llz_reconstruction_squared_error', 'llz_total_squared_error',
        'llz_code_squared_error', 'llz_max_code_error')
    for key in scalar_keys:
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid adaptive diagnostic: {key}')
    arrays = {}
    for key in ('adaptive_candidate_p', 'adaptive_candidate_payload_bytes', 'adaptive_candidate_message_bytes',
                'adaptive_candidate_squared_error', 'adaptive_candidate_total_squared_error'):
        value = metrics.get(key)
        if (not isinstance(value, list) or len(value) != 3
                or any(type(v) is not float or not math.isfinite(v) or v < 0 for v in value)):
            raise ValueError(f'Invalid adaptive candidate diagnostic: {key}')
        arrays[key] = value
    errors, costs = arrays['adaptive_candidate_squared_error'], arrays['adaptive_candidate_message_bytes']
    payloads = arrays['adaptive_candidate_payload_bytes']
    norm = metrics['qsgd_input_squared_norm']
    feasible = [i for i in adaptive_llz.CANDIDATES if adaptive_llz.admissible(errors[i], norm, budget)]
    if (arrays['adaptive_candidate_p'] != [0., 1., 2.] or errors[0] != 0
            or any(v != int(v) for v in costs + payloads)
            or any(c < b for c,b in zip(costs, payloads))
            or type(metrics.get('adaptive_selected_p')) is not int or metrics['adaptive_selected_p'] != p
            or metrics['adaptive_budget'] != budget
            or metrics['llz_max_code_error'] > p
            or (p == 0 and metrics['llz_code_squared_error'] != 0)
            or not feasible or p != min(feasible, key=lambda i: (costs[i], i))
            or metrics['llz_reconstruction_squared_error'] != errors[p]
            or metrics['llz_total_squared_error'] != arrays['adaptive_candidate_total_squared_error'][p]
            or metrics['adaptive_selected_message_bytes'] != costs[p]):
        raise ValueError('Adaptive selection or reported distortion is inconsistent')
    packets = content['qsgd']
    if (not isinstance(packets, ConfigRecord) or any(not isinstance(v, bytes) for v in packets.values())
            or sum(map(len, packets.values())) != payloads[p]):
        raise ValueError('Adaptive payload byte count mismatch')
    fixed = RecordDict(dict(content))
    fixed['update'] = ConfigRecord(update_metadata(method='qsgd-llz', levels=levels,
        server_round=server_round, llz_p=p, llz_window=window))
    return decode_update(fixed, reference, levels=levels, server_round=server_round,
                         method='qsgd-llz', llz_p=p, llz_window=window)
