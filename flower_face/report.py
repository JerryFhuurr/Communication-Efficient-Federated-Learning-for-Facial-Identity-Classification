"""Audit completed study artifacts and export tables, Markdown and PNG/SVG plots."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from flower_face.compression_study import summarize, run_label, _verify_completed_runs


def generate(directory):
    directory = Path(directory).resolve()
    record = json.loads((directory/'study.json').read_text())
    if record['status'] != 'completed':
        raise ValueError('Reporting requires a completed study')
    protocol = record['protocol']
    runs = _verify_completed_runs(record, protocol)
    summary = summarize(runs, protocol['seeds'], protocol['llz_p_values'])
    output = directory/'report'
    output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    rows, curves = [], []
    for run in runs:
        label = run_label(run)
        rows.append(dict(method=label, seed=run['config']['seed'],
            selected_round=run['best']['step'],
            selected_accuracy=run['best']['validation_accuracy'],
            selected_loss=run['best']['validation_loss'],
            upload_bits=8*run['train_upload_bytes'], total_bits=8*run['total_bytes']))
        history = [json.loads(line) for line in (Path(run['output'])/'history.jsonl').read_text().splitlines()]
        for h in history:
            curves.append(dict(method=label, seed=run['config']['seed'], round=h['round'],
                validation_accuracy=h['validation']['eval_acc'], validation_loss=h['validation']['eval_loss'],
                total_bits=h['cumulative_communication']['total']['serialized_object_bits'],
                upload_bits=h['cumulative_communication']['train']['uplink']['serialized_object_bits'],
                qsgd_error=h['train'].get('qsgd_relative_squared_error'),
                llz_error=h['train'].get('llz_relative_squared_error')))
    for name, values in [('runs', rows), ('curves', curves)]:
        with (output/f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = ['# Compression study report', '', summary['measurement'], '',
             'Values are means across seeds. MB uses 1,000,000 bytes. Selected checkpoints minimize validation loss.', '',
             '| Method | Selected accuracy | Selected loss | Upload MB | Total MB |',
             '|---|---:|---:|---:|---:|']
    for method, result in summary['methods'].items():
        lines.append(f"| {method} | {result['selected_validation_accuracy']['mean']:.2%} | "
                     f"{result['selected_validation_loss']['mean']:.4f} | "
                     f"{result['training_upload_bytes']['mean']/1e6:.3f} | "
                     f"{result['total_serialized_bytes']['mean']/1e6:.3f} |")
    lines += ['', 'Distortion curves average per-round ratios across seeds. QSGD and LLZ use different reference norms; their ratios must not be added.',
              'Positive-p runs follow different model trajectories. Thirty rounds on a small validation split do not establish converged or statistically equivalent accuracy.',
              '', 'Test data unused. CSV files contain per-seed values. All study artifacts were reconciled before export.']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = [('accuracy', 'round', 'validation_accuracy'), ('loss', 'round', 'validation_loss'),
             ('accuracy-bits', 'total_bits', 'validation_accuracy'),
             ('qsgd-distortion', 'round', 'qsgd_error'), ('llz-distortion', 'round', 'llz_error')]
    for name, xkey, ykey in plots:
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        for method in summary['methods']:
            points = [p for p in curves if p['method'] == method and p[ykey] is not None]
            if not points:
                continue
            rounds = sorted({p['round'] for p in points})
            groups = [[p for p in points if p['round'] == r] for r in rounds]
            xs = [statistics.mean(p[xkey] for p in group) for group in groups]
            ys = [statistics.mean(p[ykey] for p in group) for group in groups]
            sd = [statistics.stdev(p[ykey] for p in group) if len(group)>1 else 0 for group in groups]
            ax.plot(xs, ys, label=method, linestyle='--' if method.endswith('p0') else '-')
            ax.fill_between(xs, [y-s for y,s in zip(ys,sd)], [y+s for y,s in zip(ys,sd)], alpha=.1)
        ax.set(xlabel=xkey.replace('_',' '), ylabel=ykey.replace('_',' '), title='Mean across seeds; shading = sample SD')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        for ext in ('png','svg'):
            fig.savefig(output/f'{name}.{ext}', dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    labels = list(summary['methods'])
    ax.bar(range(len(labels)), [summary['methods'][m]['total_serialized_bytes']['mean']/1e6 for m in labels], label='Total serialized MB')
    ax.bar(range(len(labels)), [summary['methods'][m]['training_upload_bytes']['mean']/1e6 for m in labels], label='Training upload MB (included in total)')
    ax.set_xticks(range(len(labels)), labels, rotation=15)
    ax.set_ylabel('MB'); ax.legend()
    for ext in ('png','svg'):
        fig.savefig(output/f'communication.{ext}', dpi=150)
    plt.close(fig)
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    generate(parser.parse_args().directory)
