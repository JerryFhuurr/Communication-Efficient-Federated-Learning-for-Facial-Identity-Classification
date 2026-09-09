"""Plot saved learning-rate or weight-decay sweeps without launching training."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=170, facecolor='white')
    fig.savefig(path.with_suffix('.svg'), facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--parameter', choices=['learning-rate', 'weight-decay'], default='learning-rate')
    args = parser.parse_args()
    record = json.loads((args.directory/'sweep.json').read_text())
    if record['status'] != 'completed':
        raise RuntimeError('Wait for all planned runs before plotting the completed comparison')
    data = json.loads((args.directory/'curves.json').read_text())['curves']
    field = args.parameter.replace('-', '_')
    collection = 'rates' if args.parameter == 'learning-rate' else 'decays'
    label = args.parameter.replace('-', ' ').capitalize()
    seeds, rates = record['protocol']['seeds'], record['protocol'][collection]
    colors = dict(zip(rates, plt.get_cmap('tab10').colors))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .18, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(len(seeds), 3, figsize=(14, 9.4), sharex=True, sharey='col', squeeze=False)
    metrics = ('optimization_loss', 'validation_loss', 'validation_accuracy')
    titles = ('Local optimization loss', 'Validation loss', 'Validation accuracy')
    for row, seed in enumerate(seeds):
        for lr in rates:
            run = next(c for c in data if c['seed']==seed and c[field]==lr)
            history = run['history']
            selected = next(h for h in history if h['step']==run['best']['step'])
            for col, metric in enumerate(metrics):
                ax = axes[row, col]
                ax.plot([h['step'] for h in history], [h[metric] for h in history], color=colors[lr],
                        linewidth=1.1, alpha=.85, label=f'{label} {lr:g}')
                if col:
                    ax.scatter(selected['step'], selected[metric], color=colors[lr], edgecolors='white',
                               linewidth=.7, s=45, zorder=5)
                ax.set_xlim(0, record['protocol']['rounds'])
                if row==0:
                    ax.set_title(titles[col])
                if row==len(seeds)-1:
                    ax.set_xlabel('Federated round')
        axes[row, 0].set_ylabel(f'Seed {seed}\nCross-entropy')
    for col, metric in enumerate(metrics):
        if metric=='validation_accuracy':
            axes[0, col].set_ylim(0, 1)
            for ax in axes[:, col]:
                ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            axes[0, col].set_ylim(0, max(h[metric] for c in data for h in c['history'])*1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=len(rates), frameon=False)
    title = f'Uncompressed FedAvg: controlled {args.parameter} comparison'
    if args.parameter == 'weight-decay':
        title += f' (learning rate {record["protocol"]["learning_rate"]:g})'
    fig.suptitle(title, fontsize=15, y=.985)
    fig.text(.5, .023, 'Same model, split and initialization per seed. Dots: minimum-validation-loss checkpoints. Raw curves; test set unused.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .90))
    save(fig, args.directory/f'{args.parameter}-curves')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    specs = [('mean_selected_loss', 'sample_sd_selected_loss', 'Selected validation loss', 'loss'),
             ('mean_selected_accuracy', 'sample_sd_selected_accuracy', 'Selected validation accuracy', 'accuracy'),
             ('mean_final_train_accuracy', None, 'Final global training accuracy', 'train')]
    for ax, (mean_key, sd_key, title, metric) in zip(axes, specs):
        for index, lr in enumerate(rates):
            aggregate = next(r for r in record['summary'][collection] if r[field]==lr)
            for seed_index, seed in enumerate(seeds):
                offset = .2 * (seed_index / max(1, len(seeds)-1) - .5)
                run = next(c for c in data if c['seed']==seed and c[field]==lr)
                value = (run['checkpoint_evaluation']['final']['train']['accuracy'] if metric=='train'
                         else run['best']['validation_'+metric])
                ax.scatter(index+offset, value, color=colors[lr], s=28, alpha=.65)
            ax.errorbar(index, aggregate[mean_key], yerr=aggregate[sd_key] if sd_key else None,
                        marker='D', color='#222222', capsize=4, markersize=6, linewidth=1.2)
        ax.set_xticks(range(len(rates)), [f'{lr:g}' for lr in rates])
        ax.set_xlabel(label)
        ax.set_title(title, fontsize=11)
        if metric!='loss':
            ax.set_ylim(0, 1.03)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            maximum = max([c['best']['validation_loss'] for c in data] +
                          [r[mean_key] + r[sd_key] for r in record['summary'][collection]])
            ax.set_ylim(0, maximum * 1.05)
    fig.suptitle(f'{len(seeds)}-seed results at the fixed {record["protocol"]["rounds"]}-round budget', fontsize=15, y=.99)
    fig.text(.5, .025, 'Colored points: individual seeds. Black diamonds: mean. Error bars: sample SD (not confidence intervals).',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .91))
    save(fig, args.directory/f'{args.parameter}-summary')
    print(f'Saved {args.parameter} PNG/SVG figures to {args.directory.resolve()}')


if __name__=='__main__':
    main()
