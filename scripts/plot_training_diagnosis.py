"""Render static, unsmoothed learning curves from diagnose_training.py output."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

COLORS = {'none': '#4c78a8', 'qsgd': '#e07b24', 'centralized': '#218568'}
LABELS = {'none': 'Uncompressed FedAvg', 'qsgd': 'QSGD (LLZ p=0 identical)', 'centralized': 'Centralized'}


def save(fig, directory, name):
    fig.savefig(directory/f'{name}.png', dpi=170, facecolor='white')
    fig.savefig(directory/f'{name}.svg', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    data = json.loads((args.directory/'curves.json').read_text())
    curves = data['curves']
    seeds = sorted({c['seed'] for c in curves})
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .18, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(len(seeds), 3, figsize=(14, 9.4), sharex=True, sharey='col', squeeze=False)
    titles = ['Optimization loss (different training trajectories)', 'Validation loss', 'Validation accuracy']
    for row, seed in enumerate(seeds):
        for curve in [c for c in curves if c['seed'] == seed]:
            history, method = curve['history'], curve['method']
            for col, key in enumerate(('optimization_loss', 'validation_loss', 'validation_accuracy')):
                ax = axes[row, col]
                ax.plot([h['step'] for h in history], [h[key] for h in history],
                    color=COLORS[method], label=LABELS[method], linewidth=1.15, alpha=.85,
                    linestyle='--' if method == 'qsgd' else '-')
                if col > 0:
                    selected = next(h for h in history if h['step'] == curve['best']['step'])
                    ax.scatter(selected['step'], selected[key], color=COLORS[method],
                               edgecolors='white', linewidth=.7, s=42, zorder=4)
                ax.set_xlim(0, 300)
                if col == 2:
                    ax.set_ylim(0, 1)
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
                else:
                    ax.set_ylim(bottom=0)
                if row == 0:
                    ax.set_title(titles[col], fontsize=10.5)
                if row == len(seeds)-1:
                    ax.set_xlabel('Epoch / round (200 training examples)')
        axes[row, 0].set_ylabel(f'Seed {seed}\nCross-entropy')
    # Set shared limits after all methods are drawn so later curves cannot be
    # clipped by limits established from an earlier method.
    for col, key in enumerate(('optimization_loss', 'validation_loss')):
        maximum = max(h[key] for c in curves for h in c['history'])
        axes[0, col].set_ylim(0, maximum * 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .946), ncol=3, frameon=False)
    fig.suptitle('Same CNN and data exposure: centralized vs federated training', fontsize=16, y=.987)
    fig.text(.5, .024, 'Raw curves; dots mark minimum-validation-loss checkpoints. Fixed split: 200 train / 40 validation images. Test set unused.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .90), h_pad=1.5, w_pad=1.5)
    name = 'learning-curves' if data['status'] == 'completed' else 'federated-learning-curves'
    save(fig, args.directory, name)
    central = [c for c in curves if c['method'] == 'centralized']
    if central:
        fig, axes = plt.subplots(2, len(seeds), figsize=(13.5, 7.2), sharex=True, squeeze=False)
        for col, curve in enumerate(sorted(central, key=lambda c: c['seed'])):
            history = curve['history']
            for split, color in (('train', '#218568'), ('validation', '#b94b68')):
                for row, metric in enumerate(('loss', 'accuracy')):
                    axes[row, col].plot([h['step'] for h in history], [h[f'{split}_{metric}'] for h in history],
                        color=color, label=split.capitalize(), linewidth=1.2)
                    axes[row, col].axvline(curve['best']['step'], color='#666666', linestyle=':', linewidth=1)
            axes[0, col].set_title(f'Seed {curve["seed"]} | selected epoch {curve["best"]["step"]}')
            axes[0, col].set_ylim(bottom=0)
            axes[1, col].set_ylim(0, 1.02)
            axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))
            axes[1, col].set_xlabel('Centralized epoch')
        axes[0, 0].set_ylabel('Cross-entropy')
        axes[1, 0].set_ylabel('Accuracy')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
        fig.suptitle('Centralized training: fitting the training set vs validation performance', fontsize=15, y=.985)
        fig.text(.5, .025, 'Dotted lines: minimum-validation-loss checkpoints. Train metrics use the whole training set after each epoch.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, .89))
        save(fig, args.directory, 'centralized-generalization')
    (args.directory/'plot-metadata.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,
        source_curves_status=data['status'], smoothing=False, formats=['png', 'svg']), indent=2)+'\n')
    print(f'Saved figures to {args.directory.resolve()}')


if __name__ == '__main__':
    main()
