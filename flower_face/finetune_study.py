"""Bounded uncompressed fine-tuning: two seed-42 settings, conditional confirmation."""
import argparse
from contextlib import contextmanager
import ctypes
import json
import math
import os
from pathlib import Path
import statistics
import time

from flower_face.finetune_task import file_hash, read_manifest
from flower_face.run import launch
from flower_face.study import collect, snapshot, write_json
from flower_face.reproducibility import source_hash


@contextmanager
def keep_awake():
    """Prevent automatic Windows idle sleep only while this batch is running."""
    enabled = os.name == 'nt' and ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        if enabled:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def run_batch(manifest_path, output, baseline_path, rounds=15, rates=(0.001,0.003), resume=False):
    if rounds < 1 or len(rates) != 2 or len(set(rates)) != 2 or any(not math.isfinite(lr) or lr <= 0 for lr in rates):
        raise ValueError('Specify positive rounds and exactly two distinct positive learning rates')
    root = Path(__file__).resolve().parents[1]
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    baseline_path = Path(baseline_path).resolve()
    baseline = json.loads((baseline_path/'metrics.json').read_text())
    manifest_raw = json.loads(manifest_path.read_text())
    classes, clients = len(manifest_raw['identities']), manifest_raw['num_clients']
    manifest = read_manifest(manifest_path, classes, clients)
    if (baseline['config']['seed'] != 42 or baseline['config']['compression'] != 'none'
            or baseline['config']['task-module'] != 'flower_face.pretrained_task'
            or baseline['test'] is not None or 'initial_head' not in manifest
            or manifest['initial_head']['sha256'] != file_hash(baseline_path/'best_model.pt')
            or manifest['feature_extractor']['source_feature_manifest_sha256'] != baseline['manifest_sha256']):
        raise ValueError('Baseline must match the validation-only seed-42 warm start')
    config = dict(baseline['config'], **{'task-module':'flower_face.finetune_task',
        'manifest':manifest_path.as_posix(), 'num-server-rounds':rounds})
    if resume:
        record=json.loads((output/'study.json').read_text())
        if record['status'] not in ('failed','completed'):
            raise ValueError('Resume requires a stopped batch; inspect any running job first')
        protocol=record['protocol']
        if (protocol['rates'] != list(rates) or protocol['rounds'] != rounds
                or protocol['manifest_sha256'] != file_hash(manifest_path)
                or protocol['source_sha256'] != source_hash(output/'app')
                or record['baseline_output'] != str(baseline_path)
                or record['baseline'] != baseline['best_checkpoint']):
            raise ValueError('Resume configuration or frozen source changed')
        if record['status']=='completed':
            print(f'Already complete: {output}',flush=True)
            return
        record.setdefault('interruptions',[]).append(record.pop('error','Stopped batch'))
        for attempt in record['attempts']:
            if attempt['status']=='running':
                attempt['status']='interrupted'
        record['status']='running'
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot(root,output/'app')
        record = dict(status='running', runs=[], attempts=[], baseline_output=str(baseline_path),
        baseline=baseline['best_checkpoint'], protocol=dict(rates=list(rates),rounds=rounds,
            pilot_seed=42,confirmation_seeds=[43,44],classifier_lr_multiplier=10,
            selection='Lowest validation loss; confirm only if lower than warm-start baseline',
            source_sha256=source_hash(output/'app'),manifest_sha256=file_hash(manifest_path),
            shared_warm_start=manifest['initial_head'],test_evaluated=False))

    def one(seed,lr,label):
        for saved in record['runs']:
            if saved['config']['seed']==seed and saved['config']['learning-rate']==lr:
                refreshed=collect(Path(saved['output']).parent,saved['config'])
                if refreshed['best'] != saved['best'] or refreshed['replay'] != saved['replay']:
                    raise ValueError('Previously completed result changed')
                print(f'Reusing completed {label}',flush=True)
                return saved
        count=sum(a['seed']==seed and a['lr']==lr for a in record['attempts'])
        folder=output/(label if count==0 else f'{label}-attempt-{count+1}')
        folder.mkdir()
        cfg=dict(config,seed=seed,**{'learning-rate':lr,'output-dir':folder.as_posix()})
        attempt=dict(seed=seed,lr=lr,output=str(folder),status='running')
        record['attempts'].append(attempt)
        write_json(output/'study.json',record)
        print(f'Starting {label}: {rounds} rounds, layer4 LR={lr}, head LR={10*lr}',flush=True)
        start=time.perf_counter()
        if launch(cfg,root=root,app_dir=output/'app',log_path=folder/'terminal.log'):
            raise RuntimeError(f'Run failed: {folder / "terminal.log"}')
        result=collect(folder,cfg)
        assert result['metadata']['source_sha256']==record['protocol']['source_sha256']
        assert result['metadata']['manifest_sha256']==record['protocol']['manifest_sha256']
        if record['runs']:
            assert result['metadata']['initial_model_sha256']==record['runs'][0]['metadata']['initial_model_sha256']
        result.update(label=label,elapsed_seconds=time.perf_counter()-start)
        record['runs'].append(result)
        attempt['status']='completed'
        write_json(output/'study.json',record)
        print(f"Completed {label}: {result['best']}",flush=True)
        return result

    try:
        candidates=[one(42,lr,f'pilot-lr-{lr:g}') for lr in rates]
        selected=min(candidates,key=lambda r:r['best']['validation_loss'])
        record['selected_learning_rate']=selected['config']['learning-rate']
        record['selected_checkpoint']=str(Path(selected['output'])/'best_model.pt')
        record['improved_validation_loss']=selected['best']['validation_loss'] < record['baseline']['validation_loss']
        if record['improved_validation_loss']:
            for seed in (43,44):
                one(seed,record['selected_learning_rate'],f'confirmation-seed-{seed}')
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=str(error))
        raise
    finally:
        write_json(output/'study.json',record)
    report(output,record)
    print(f'Completed bounded batch and artifact audit: {output}',flush=True)


def report(output,record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in record['runs'] if r['config']['learning-rate']==record['selected_learning_rate']]
    values=[r['best']['validation_accuracy'] for r in selected]
    record['selected_accuracy_mean']=statistics.mean(values)
    record['selected_accuracy_sample_sd']=statistics.stdev(values) if len(values)>1 else None
    write_json(output/'study.json',record)
    lines=['# Final-stage fine-tuning batch','','Fixed data split, four IID clients. ResNet layers through layer3 are frozen; layer4 and the linear head are trained. BatchNorm running statistics stay fixed. The classifier learning rate is 10 times the listed layer4 rate. All runs start from the same validation-selected seed-42 head and original ImageNet layer4.','','| Run | Layer4 LR | Selected round | Validation accuracy | Validation loss | Total serialized GB | Minutes |','|---|---:|---:|---:|---:|---:|---:|']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for r in record['runs']:
        b=r['best']
        lines.append(f"| {r['label']} | {r['config']['learning-rate']} | {b['step']} | {100*b['validation_accuracy']:.2f}% | {b['validation_loss']:.4f} | {r['total_bytes']/1e9:.3f} | {r['elapsed_seconds']/60:.1f} |")
        rounds=[v['round'] for v in r['replay']]
        axes[0].plot(rounds,[100*v['validation']['eval_acc'] for v in r['replay']],label=r['label'])
        axes[1].plot(rounds,[v['validation']['eval_loss'] for v in r['replay']],label=r['label'])
    for axis,metric,label in zip(axes,('validation_accuracy','validation_loss'),('Validation accuracy (%)','Validation loss')):
        value=record['baseline'][metric]*(100 if metric=='validation_accuracy' else 1)
        axis.axhline(value,linestyle='--',color='black',label='Warm-start baseline')
        axis.set(xlabel='Additional federated round',ylabel=label)
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'learning-curves.png',dpi=160)
    plt.close(fig)
    sd=record['selected_accuracy_sample_sd']
    lines += ['',f"Selected layer4 LR: {record['selected_learning_rate']}. Mean validation accuracy: {100*statistics.mean(values):.2f}% across {len(values)} seed(s)." + (f' Sample SD: {100*sd:.2f} percentage points.' if sd is not None else ''),'',
        'Selection minimizes validation loss. Confirmation is conditional on improving the shared warm-start baseline loss. These seeds vary fine-tuning randomness only: they share the same tuned seed-42 starting head and fixed dataset split. This is an additional-training comparison, not an equal-compute comparison with the head-only model. The held-out test is unused.', '',
        'Communication is cumulative logical serialized Flower objects, not network traffic. Prefix distribution, feature extraction, and all warm-start training are excluded; full layer4/head state including BatchNorm buffers is transferred. Compression was not tested in this batch.', '',
        'All completed runs were checked for expected configuration, full client participation, complete round history, checkpoint hashes and reconciled communication counts. study.json records the frozen source hash, shared warm-start provenance and original baseline path. Interrupted attempts, if any, remain in their original directories and are excluded from completed-run statistics and byte totals.','',
        '![Learning curves](learning-curves.png)']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=15)
    parser.add_argument('--rates',type=float,nargs=2,default=[0.001,0.003])
    parser.add_argument('--resume',action='store_true',help='Reuse completed results from a stopped batch')
    args=parser.parse_args()
    with keep_awake():
        run_batch(args.manifest,args.output,args.baseline,args.rounds,args.rates,args.resume)
