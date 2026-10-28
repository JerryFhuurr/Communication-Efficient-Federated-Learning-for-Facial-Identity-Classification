"""Whole-update LLZ selection, with no Flower, Torch or dataset dependency.

One QSGD draw per tensor is shared by p=0/1/2. The constraint is additional
LLZ squared error <= budget * original update squared norm. A caller can
provide the actual serialized-message cost; the default counts packet bytes.
"""

from dataclasses import dataclass
import math

import numpy as np

from compression import qsgd, qsgd_llz

CANDIDATES = (0, 1, 2)


@dataclass
class Candidate:
    p: int
    packets: dict
    stats: dict

    @property
    def payload_bytes(self):
        return sum(map(len, self.packets.values()))


@dataclass
class Selection:
    selected: Candidate
    candidates: tuple
    costs: tuple
    budget: float


def validate_budget(budget):
    if (type(budget) not in (int, float) or not math.isfinite(budget)
            or budget < 0):
        raise ValueError('adaptive LLZ budget must be finite and nonnegative')
    return float(budget)


def admissible(error, squared_norm, budget):
    # Avoid overflow in budget*norm and do not introduce an epsilon at zero.
    return error == 0 if squared_norm == 0 or budget == 0 else error / squared_norm <= budget


def _squared(values):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        squared = float(np.sum(values * values, dtype=np.float64))
    if not math.isfinite(squared) or (squared == 0 and np.any(values != 0)):
        raise ValueError('Update scale is unsupported for float64 squared-error measurements')
    return squared


def select(updates, *, levels, rng, budget, window_size=128, cost=None):
    """Return the smallest feasible candidate, ties broken by lower p.

    ``cost(candidate, all_candidates)`` must return an integer byte count.
    All candidates are encoded/decoded before evaluating costs, allowing the
    caller to include every candidate's diagnostics in its message framing.
    Float32/64 reconstruction is measured after the actual codec decode.
    """
    budget = validate_budget(budget)
    if not updates or any(not isinstance(name, str) for name in updates):
        raise ValueError('Expected a nonempty mapping of named update tensors')
    common = dict(qsgd_quantization_squared_error=0.0, qsgd_input_squared_norm=0.0,
                  qsgd_max_abs_error=0.0, qsgd_coordinate_count=0)
    candidates = tuple(Candidate(p, {}, dict(
        llz_code_squared_error=0.0, llz_reconstruction_squared_error=0.0,
        llz_qsgd_squared_norm=0.0, llz_max_code_error=0, llz_symbol_count=0,
        llz_total_squared_error=0.0)) for p in CANDIDATES)
    for name in sorted(updates):
        original = np.asarray(updates[name])
        q = qsgd.quantize(original, levels=levels, rng=rng)
        full = original.astype(np.float64)
        z = qsgd.dequantize(q).astype(np.float64)
        error = z - full
        common['qsgd_quantization_squared_error'] += _squared(error)
        common['qsgd_input_squared_norm'] += _squared(full)
        common['qsgd_max_abs_error'] = max(common['qsgd_max_abs_error'],
                                         float(np.max(np.abs(error), initial=0)))
        common['qsgd_coordinate_count'] += q.codes.size
        for candidate in candidates:
            packet = qsgd_llz.pack(q, p=candidate.p, window_size=window_size)
            restored = qsgd_llz.unpack(packet, max_symbols=q.codes.size,
                expected_p=candidate.p, expected_window_size=window_size)
            values = qsgd.dequantize(restored).astype(np.float64)
            code_error = restored.codes.astype(np.int64) - q.codes.astype(np.int64)
            extra, total = values - z, values - full
            stats = candidate.stats
            stats['llz_code_squared_error'] += float(np.sum(code_error * code_error))
            stats['llz_reconstruction_squared_error'] += _squared(extra)
            stats['llz_qsgd_squared_norm'] += _squared(z)
            stats['llz_total_squared_error'] += _squared(total)
            stats['llz_max_code_error'] = max(stats['llz_max_code_error'],
                int(np.max(np.abs(code_error), initial=0)))
            stats['llz_symbol_count'] += int(q.codes.size)
            candidate.packets[name] = packet
    if common['qsgd_coordinate_count'] == 0:
        raise ValueError('Update must contain at least one coordinate')
    for candidate in candidates:
        candidate.stats.update(common)
        if any(not math.isfinite(value) for value in candidate.stats.values()):
            raise ValueError('Nonfinite adaptive LLZ distortion; update scale is unsupported')
    costs = tuple(cost(c, candidates) if cost else c.payload_bytes for c in candidates)
    if any(type(size) is not int or size < 0 for size in costs):
        raise ValueError('Candidate byte costs must be nonnegative integers')
    feasible = [c for c in candidates if admissible(
        c.stats['llz_reconstruction_squared_error'], common['qsgd_input_squared_norm'], budget)]
    # p=0 is always feasible; a codec failure must raise rather than relax budget.
    if not feasible or candidates[0].stats['llz_reconstruction_squared_error'] != 0:
        raise RuntimeError('Lossless LLZ fallback failed')
    chosen = min(feasible, key=lambda c: (costs[c.p], c.p))
    return Selection(chosen, candidates, costs, budget)
