"""Audit saved feasibility artifacts and report without launching experiments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def report(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to((Path.cwd() / 'outputs').resolve()):
        raise ValueError('Reports must stay under outputs/')
    summary = json.loads((directory / 'summary.json').read_text())
    rows = json.loads((directory / 'probes.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    if summary['status'] != 'completed' or len(rows) != 120:
        raise ValueError('Complete four-client batch required')
    for client in summary['fits']:
        for name in ['model', 'decoder']:
            payload = (directory / f"client-{client['client']}/{name}.aew").read_bytes()
            if hashlib.sha256(payload).hexdigest() != client[f'{name}_sha256']:
                raise ValueError('Saved weight artifact checksum mismatch')
        if client['selected_validation_mse'] != min(h['validation_normalized_masked_mse'] for h in client['history']):
            raise ValueError('Checkpoint selection mismatch')
    if len({(r['snapshot'], r['method']) for r in rows}) != 120:
        raise ValueError('Duplicate or missing probe results')
    methods = ['autoencoder', 'qsgd', 'topk']
    for method in methods:
        subset = [r for r in rows if r['method'] == method]
        saved = summary['methods'][method]
        if len(subset) != 40 or sum(r['packet_bytes'] for r in subset) != saved['total_probe_bytes']:
            raise ValueError('Probe byte total mismatch')
        if any(r['signature'] != summary['signature'] or not np.isfinite(r['relative_squared_error']) for r in subset):
            raise ValueError('Probe signature or numeric mismatch')
    for snapshot in {r['snapshot'] for r in rows}:
        subset = [r for r in rows if r['snapshot'] == snapshot]
        if len({r['energy'] for r in subset}) != 1 or len({r['snapshot_sha256'] for r in subset}) != 1:
            raise ValueError('Comparators did not use the same original update')
    lines = ['# Standalone learned-codec feasibility results', '',
        'Completed the fixed Wednesday fitting and Thursday reconstruction batch early. '
        'Four independent client autoencoders, ten epochs each; 40 full held-out updates, '
        'three codecs per update. No Flower training or identity accuracy evaluation.', '',
        '| Codec | Bytes/update | Energy-weighted relative squared error | Mean cosine | Median encode s | Median decode s |',
        '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        m = summary['methods'][method]
        lines.append(f"| {method} | {m['packet_bytes_per_update']:,} | {m['energy_weighted_relative_squared_error']:.6f} | {m['mean_cosine']:.6f} | {m['median_encode_seconds']:.4f} | {m['median_decode_seconds']:.4f} |")
    lines += ['', 'Relative squared error is total squared coordinate error divided by total '
        'original update energy, pooled over all 40 updates. Padding is excluded. '
        'It is not classification error or percentage-point accuracy loss. '
        'Cosine measures direction; four client trajectories are not four independent FL seed runs.', '',
        '## Fitting and setup', '',
        '| Client | Selected epoch | Sampled normalized masked validation MSE | Fitting seconds |',
        '|---|---:|---:|---:|']
    for client in summary['fits']:
        lines.append(f"| {client['client']} | {client['selected_epoch']} | {client['selected_validation_mse']:.8f} | {client['fitting_seconds']:.3f} |")
    fitting_time = sum(c['fitting_seconds'] for c in summary['fits'])
    lines += ['', f"Calibration collection: {summary['calibration_seconds']:.3f} s. "
        f"Sequential AE fitting: {fitting_time:.3f} s. "
        f"Current batch wall time: {summary['elapsed_seconds']:.3f} s, including reading/hashing and probes. "
        'These are additional compute costs, not included in earlier FL baselines.', '',
        f"Four decoder setup artifacts: **{summary['decoder_setup_bytes_four_clients']:,} bytes**. "
        'Encoder artifacts remain local. Calibration updates were stored locally for this simulation; '
        'no calibration-upload protocol was measured. Zero calibration traffic is an assumption of '
        'local fitting, not a measurement of this centralized offline runner.', '',
        '## Communication accounting', '',
        'AE packets include the version header, layout, decoder identity, float32 scales and latents. '
        'QSGD probes quantize each tensor separately at 127 levels, as in the existing baseline, '
        'with a fixed seeded draw per update. Their offline bundle includes layout and '
        'per-tensor length framing. Top-k retains 124,000 coordinates globally, without feedback, '
        'and includes its index/value packet plus layout framing. '
        'These are actual constructed codec bundle lengths, not Flower serialized messages or network traffic.', '',
        '| Codec | Projected bytes for 60 uploads + decoder setup |', '|---|---:|']
    for method, value in summary['codec_only_15_round_projection'].items():
        lines.append(f'| {method} | {value:,} |')
    lines += ['', 'Projection assumes the same tensor shapes for four clients and 15 rounds, '
        'one decoder distribution and no refresh. Full-model downloads, fixed buffers, Flower/control '
        'records and any remote calibration traffic are excluded. This is not a new 15-round FL result.', '',
        f"Codec-only strict byte break-even rounds: {summary['codec_only_break_even_rounds']}. "
        'This counts decoder distribution but does not establish wall-time or end-to-end break-even.', '',
        '## Per-client full-update reconstruction', '',
        '| Client | AE relative squared error | QSGD relative squared error | Top-k relative squared error |',
        '|---|---:|---:|---:|']
    for client in range(4):
        values = []
        for method in methods:
            subset = [r for r in rows if r['client'] == client and r['method'] == method]
            values.append(sum(r['squared_error'] for r in subset) / sum(r['energy'] for r in subset))
        lines.append(f"| {client} | " + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', '## Interpretation and literature', '',
        '[Chandar et al. (2021)](https://arxiv.org/abs/2108.05670) establishes a precedent '
        'for learned update compression with client encoders and aggregation-node decoders. '
        'Our blockwise normalized-delta model differs from their method; their compression '
        'ratios and accuracy findings cannot be transferred to this experiment.', '',
        '[Stich et al. (2018)](https://arxiv.org/abs/1809.07599) studies sparsification with '
        'memory/error compensation. Here Top-k has no feedback because each held-out update '
        'is an independent codec probe. These results cannot predict the earlier Top-k+EF '
        'training result or establish a FedAvg convergence guarantee.', '',
        'The fitting objective weights sampled blocks by represented population and masks padding. '
        'Checkpoint selection uses sampled normalized reconstruction-validation MSE; full-update '
        'errors are measured in original coordinate units and may rank models differently. '
        'Validation updates are distinct trajectories on the same training-image population and '
        'same initialization. They do not cover later-round update drift or unseen clients. '
        'The reconstruction-validation trajectory is used for checkpoint selection, so it is '
        'a development set, not an untouched test set.', '',
        'The final proceed/stop assessment is recorded separately in `decision.md` when completed. Useful packet savings alone '
        'are insufficient: reconstruction error, direction, compute/setup cost and likely update '
        'drift must be considered. No additional architecture tuning or FL integration was launched.', '',
        '## Audit', '',
        'Saved weight hashes, minimum-loss selection, 120 distinct probe records, byte totals, '
        'shared original-update hashes/energies and finite error values passed the saved-artifact audit. '
        'Protocol, source hashes, per-epoch histories, per-tensor errors and fixed RNG seeds are stored '
        'alongside this report. Generated outputs remain excluded from Git.', '']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib unavailable; numerical report saved.')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for method in methods:
            subset = [r for r in rows if r['method'] == method]
            axes[0].plot(range(1, 41), [r['relative_squared_error'] for r in subset], label=method)
        axes[0].set(xlabel='Held-out update (ordered by client, then step)', ylabel='Relative squared error',
                    title='Full-update reconstruction')
        axes[0].legend()
        axes[1].bar(methods, [summary['methods'][m]['packet_bytes_per_update'] / 1e6 for m in methods])
        axes[1].set(ylabel='Codec bundle MB/update', title='Includes packet metadata; excludes setup')
        fig.savefig(directory / 'reconstruction.png', dpi=180)
        plt.close(fig)
    print(f'Saved report: {directory / "report.md"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    report(parser.parse_args().directory)
