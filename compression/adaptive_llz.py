"""Choose one LLZ tolerance for a complete client update.

Each tensor is quantized once. The LLZ candidates share those QSGD codes,
and the distortion budget applies to the sum across all tensors.
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
    if squared_norm == 0 or budget == 0:
        return error == 0
    # Divide to avoid overflowing when the update norm and budget are large.
    return error / squared_norm <= budget


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
    qsgd_stats = dict(
        qsgd_quantization_squared_error=0.0,
        qsgd_input_squared_norm=0.0,
        qsgd_max_abs_error=0.0,
        qsgd_coordinate_count=0,
    )
    candidates = tuple(Candidate(p, {}, dict(
        llz_code_squared_error=0.0,
        llz_reconstruction_squared_error=0.0,
        llz_qsgd_squared_norm=0.0,
        llz_max_code_error=0,
        llz_symbol_count=0,
        llz_total_squared_error=0.0,
    )) for p in CANDIDATES)
    for name in sorted(updates):
        original = np.asarray(updates[name])
        quantized = qsgd.quantize(original, levels=levels, rng=rng)
        original_values = original.astype(np.float64)
        qsgd_values = qsgd.dequantize(quantized).astype(np.float64)
        quantization_error = qsgd_values - original_values
        symbol_count = quantized.codes.size
        qsgd_stats['qsgd_quantization_squared_error'] += _squared(quantization_error)
        qsgd_stats['qsgd_input_squared_norm'] += _squared(original_values)
        qsgd_stats['qsgd_max_abs_error'] = max(
            qsgd_stats['qsgd_max_abs_error'],
            float(np.max(np.abs(quantization_error), initial=0)),
        )
        qsgd_stats['qsgd_coordinate_count'] += symbol_count

        for candidate in candidates:
            packet = qsgd_llz.pack(quantized, p=candidate.p, window_size=window_size)
            restored = qsgd_llz.unpack(
                packet, max_symbols=symbol_count, expected_p=candidate.p,
                expected_window_size=window_size,
            )
            restored_values = qsgd.dequantize(restored).astype(np.float64)
            code_error = restored.codes.astype(np.int64) - quantized.codes.astype(np.int64)
            # The budget uses the error added to the QSGD reconstruction.
            added_error = restored_values - qsgd_values
            total_error = restored_values - original_values
            stats = candidate.stats
            stats['llz_code_squared_error'] += float(np.sum(code_error * code_error))
            stats['llz_reconstruction_squared_error'] += _squared(added_error)
            stats['llz_qsgd_squared_norm'] += _squared(qsgd_values)
            stats['llz_total_squared_error'] += _squared(total_error)
            stats['llz_max_code_error'] = max(
                stats['llz_max_code_error'], int(np.max(np.abs(code_error), initial=0)),
            )
            stats['llz_symbol_count'] += int(symbol_count)
            candidate.packets[name] = packet

    if qsgd_stats['qsgd_coordinate_count'] == 0:
        raise ValueError('Update must contain at least one coordinate')
    for candidate in candidates:
        candidate.stats.update(qsgd_stats)
        if any(not math.isfinite(value) for value in candidate.stats.values()):
            raise ValueError('Nonfinite adaptive LLZ distortion; update scale is unsupported')

    costs = tuple(
        cost(candidate, candidates) if cost else candidate.payload_bytes
        for candidate in candidates
    )
    if any(type(size) is not int or size < 0 for size in costs):
        raise ValueError('Candidate byte costs must be nonnegative integers')

    update_norm = qsgd_stats['qsgd_input_squared_norm']
    feasible = [
        candidate for candidate in candidates
        if admissible(candidate.stats['llz_reconstruction_squared_error'], update_norm, budget)
    ]
    # If p=0 adds error, the lossless codec has failed.
    if not feasible or candidates[0].stats['llz_reconstruction_squared_error'] != 0:
        raise RuntimeError('Lossless LLZ fallback failed')
    chosen = min(feasible, key=lambda candidate: (costs[candidate.p], candidate.p))
    return Selection(chosen, candidates, costs, budget)
