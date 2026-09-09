"""Plot a completed training-only augmentation sweep from saved results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

COLORS = {'none':'#4c78a8', 'horizontal-flip':'#e07b24'}
LABELS = {'none':'No augmentation', 'horizontal-flip':'Random horizontal flip (p=0.5)'}


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=170, facecolor='white')
    fig.savefig(path.with_suffix('.svg'), facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    record = json.loads((args.directory/'sweep.json').read_text())
    if record['status'] != 'completed':
        raise RuntimeError('Wait for the completed sweep before plotting')
    curves = json.loads((args.directory/'curves.json').read_text())['curves']
    seeds, methods = record['protocol']['seeds'], record['protocol']['augmentations']
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.grid':True, 'grid.alpha':.18, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(len(seeds), 3, figsize=(14,9.4), sharex=True, sharey='col', squeeze=False)
    metrics = ('optimization_loss','validation_loss','validation_accuracy')
    titles = ('Local optimization loss','Validation loss','Validation accuracy')
    for row, seed in enumerate(seeds):
        for method in methods:
            run = next(c for c in curves if c['seed']==seed and c['augmentation']==method)
            selected = next(h for h in run['history'] if h['step']==run['best']['step'])
            for col, metric in enumerate(metrics):
                ax=axes[row,col]
                ax.plot([h['step'] for h in run['history']], [h[metric] for h in run['history']],
                        color=COLORS[method], label=LABELS[method], linewidth=1.15, alpha=.85)
                if col:
                    ax.scatter(selected['step'], selected[metric], color=COLORS[method],
                               edgecolors='white', linewidth=.7, s=45, zorder=5)
                ax.set_xlim(0,record['protocol']['rounds'])
                if row==0: ax.set_title(titles[col])
                if row==len(seeds)-1: ax.set_xlabel('Federated round')
        axes[row,0].set_ylabel(f'Seed {seed}\nCross-entropy')
    for col, metric in enumerate(metrics):
        if metric=='validation_accuracy':
            axes[0,col].set_ylim(0,1)
            for ax in axes[:,col]: ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            axes[0,col].set_ylim(0,max(h[metric] for c in curves for h in c['history'])*1.05)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2,frameon=False)
    fig.suptitle('Uncompressed FedAvg: controlled training-only augmentation comparison',fontsize=15,y=.985)
    fig.text(.5,.022,'Learning rate 0.1, weight decay 0. Dots: minimum-validation-loss checkpoints. Validation/test images are never augmented.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.90))
    save(fig,args.directory/'augmentation-curves')

    summary=record['summary']['augmentations']
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    specs=(('mean_selected_loss','sample_sd_selected_loss','Selected validation loss','loss'),
           ('mean_selected_accuracy','sample_sd_selected_accuracy','Selected validation accuracy','accuracy'),
           ('mean_final_train_accuracy',None,'Final global training accuracy','train'))
    for ax,(mean_key,sd_key,title,metric) in zip(axes,specs):
        for index,method in enumerate(methods):
            aggregate=next(r for r in summary if r['augmentation']==method)
            for seed_index,seed in enumerate(seeds):
                run=next(c for c in curves if c['seed']==seed and c['augmentation']==method)
                value=(run['checkpoint_evaluation']['final']['train']['accuracy'] if metric=='train'
                       else run['best']['validation_'+metric])
                offset=.2*(seed_index/max(1,len(seeds)-1)-.5)
                ax.scatter(index+offset,value,color=COLORS[method],s=30,alpha=.65)
            ax.errorbar(index,aggregate[mean_key],yerr=aggregate[sd_key] if sd_key else None,
                        marker='D',color='#222222',capsize=4,markersize=6,linewidth=1.2)
        ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=8)
        ax.set_title(title,fontsize=11)
        if metric=='loss':
            maximum=max(r[mean_key]+r[sd_key] for r in summary)
            ax.set_ylim(0,maximum*1.05)
        else:
            ax.set_ylim(0,1.03); ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Three-seed augmentation results at the fixed 300-round budget',fontsize=15,y=.99)
    fig.text(.5,.02,'Colored points: seeds 42–44. Black diamonds: mean. Error bars: sample SD, not confidence intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.91))
    save(fig,args.directory/'augmentation-summary')
    print(f'Saved augmentation PNG/SVG figures to {args.directory.resolve()}')


if __name__=='__main__':
    main()
