"""One bounded learning-rate reduction, with conditional seed confirmation."""
import argparse
import json
from pathlib import Path
import statistics
import time

from flower_face.finetune_study import keep_awake
from flower_face.finetune_task import file_hash
from flower_face.reproducibility import model_hash, source_hash
from flower_face.run import launch
from flower_face.study import collect, snapshot, write_json
from flower_face.task_api import resolve


def promising(baseline, candidate):
    return (candidate['validation_loss'] < baseline['validation_loss']
            and candidate['validation_accuracy'] - baseline['validation_accuracy'] >= 0.005 - 1e-12)


def run_batch(previous_path, output, resume=False):
    root = Path(__file__).resolve().parents[1]
    previous_path, output = Path(previous_path).resolve(), Path(output).resolve()
    previous = json.loads(previous_path.read_text())
    if previous['status'] != 'completed':
        raise ValueError('The previous fine-tuning batch must be complete')
    bases = {r['config']['seed']: r for r in previous['runs']
             if r['config']['learning-rate'] == previous['selected_learning_rate']}
    if set(bases) != {42,43,44}:
        raise ValueError('Expected the completed fine-tuning seeds 42, 43 and 44')
    protocol = dict(previous_study=str(previous_path), previous_sha256=file_hash(previous_path),
        rounds=5, lr_factor=0.1, classifier_lr_multiplier=10, seed_offset=10000,
        pilot_seed=42, confirmation_seeds=[43,44], minimum_accuracy_gain_pp=0.5,
        gate='Lower validation loss and >=0.5 percentage-point accuracy gain on seed 42',
        promotion='Mean accuracy gain >=0.5 pp and lower mean loss; include starting checkpoints in selection',
        test_evaluated=False)
    if resume:
        record = json.loads((output/'study.json').read_text())
        if record['protocol'] != protocol or source_hash(output/'app') != record['source_sha256']:
            raise ValueError('Saved protocol or frozen source changed')
        if record['status'] == 'completed':
            print(f'Already complete: {output}', flush=True)
            return
        if record['status'] != 'failed':
            raise ValueError('Resume requires a stopped, failed batch')
        record.setdefault('interruptions', []).append(record.pop('error', 'Interrupted'))
        for attempt in record['attempts']:
            if attempt['status'] == 'running':
                attempt['status'] = 'interrupted'
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot(root, output/'app')
        record = dict(protocol=protocol, source_sha256=source_hash(output/'app'),
                      runs=[], attempts=[], baseline_results={str(s):r['best'] for s,r in bases.items()})
    record['status'] = 'running'

    def one(seed):
        for saved in record['runs']:
            if saved['source_seed'] == seed:
                fresh = collect(Path(saved['output']).parent, saved['config'])
                if any(fresh[k] != saved[k] for k in fresh):
                    raise ValueError('Completed result changed')
                return saved
        base = bases[seed]
        verified = collect(Path(base['output']).parent, base['config'])
        if any(verified[k] != base[k] for k in verified):
            raise ValueError('Starting experiment changed')
        checkpoint = Path(base['output'])/'best_model.pt'
        count = sum(a['source_seed'] == seed for a in record['attempts'])
        folder = output/f'seed-{seed}-attempt-{count+1}'
        folder.mkdir()
        config = dict(base['config'], seed=seed+10000, **{
            'num-server-rounds':5, 'learning-rate':base['config']['learning-rate']*0.1,
            'initial-checkpoint':checkpoint.as_posix(), 'initial-checkpoint-sha256':file_hash(checkpoint),
            'output-dir':folder.as_posix(), 'evaluate-final-test':False})
        expected_initial_hash = model_hash(resolve(config).Net(config['num-classes']).state_dict())
        attempt = dict(source_seed=seed, output=str(folder), status='running')
        record['attempts'].append(attempt)
        write_json(output/'study.json', record)
        print(f"Starting source seed {seed}: 5 extra rounds, LR={config['learning-rate']:g}, training seed={config['seed']}", flush=True)
        start = time.perf_counter()
        if launch(config, root=root, app_dir=output/'app', log_path=folder/'terminal.log'):
            raise RuntimeError(f'Run failed: {folder / "terminal.log"}')
        result = collect(folder, config)
        if (result['metadata']['initial_model_sha256'] != expected_initial_hash
                or result['metadata']['source_sha256'] != record['source_sha256']
                or result['metadata']['manifest_sha256'] != base['metadata']['manifest_sha256']):
            raise ValueError('Continuation initialization or provenance mismatch')
        use_candidate = result['best']['validation_loss'] < base['best']['validation_loss']
        result.update(source_seed=seed, elapsed_seconds=time.perf_counter()-start,
            starting_checkpoint=str(checkpoint), starting_result=base['best'],
            selected=result['best'] if use_candidate else base['best'],
            selected_checkpoint=str(Path(result['output'])/'best_model.pt') if use_candidate else str(checkpoint))
        record['runs'].append(result)
        attempt['status'] = 'completed'
        write_json(output/'study.json', record)
        print(f"Completed source seed {seed}: {result['best']}", flush=True)
        return result

    try:
        pilot = one(42)
        record['confirmation_gate_passed'] = promising(bases[42]['best'], pilot['best'])
        if record['confirmation_gate_passed']:
            for seed in (43,44):
                one(seed)
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        write_json(output/'study.json', record)
    report(output, record)
    print(f'Completed bounded refinement and audit: {output}', flush=True)


def report(output, record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = record['runs']
    baseline_acc = statistics.mean(r['starting_result']['validation_accuracy'] for r in runs)
    selected_acc = statistics.mean(r['selected']['validation_accuracy'] for r in runs)
    loss_improved = statistics.mean(r['selected']['validation_loss'] for r in runs) < statistics.mean(r['starting_result']['validation_loss'] for r in runs)
    record['summary'] = dict(n_seeds=len(runs), starting_accuracy_mean=baseline_acc,
        selected_accuracy_mean=selected_acc, accuracy_gain_pp=100*(selected_acc-baseline_acc),
        selected_accuracy_sample_sd=statistics.stdev(r['selected']['validation_accuracy'] for r in runs) if len(runs)>1 else None,
        promote=len(runs)==3 and loss_improved and selected_acc-baseline_acc>=0.005-1e-12)
    write_json(output/'study.json', record)
    lines=['# Bounded low-learning-rate continuation','',
        'Five additional uncompressed rounds per run; layer4 LR=0.0003 and classifier LR=0.003 for the current baseline. Architecture, data split and feature cache are unchanged. Each continuation starts from its corresponding earlier best checkpoint; new training seeds are 10042/10043/10044. The starting checkpoint remains eligible for selection by lowest validation loss.','',
        '| Source seed | Starting accuracy | Candidate accuracy | Candidate loss | Selected accuracy | Extra serialized GB |',
        '|---|---:|---:|---:|---:|---:|']
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for r in runs:
        b,c,s=r['starting_result'],r['best'],r['selected']
        lines.append(f"| {r['source_seed']} | {100*b['validation_accuracy']:.2f}% | {100*c['validation_accuracy']:.2f}% | {c['validation_loss']:.4f} | {100*s['validation_accuracy']:.2f}% | {r['total_bytes']/1e9:.3f} |")
        x=[0]+[v['round'] for v in r['replay']]
        axes[0].plot(x,[100*b['validation_accuracy']]+[100*v['validation']['eval_acc'] for v in r['replay']],marker='o',label=str(r['source_seed']))
        axes[1].plot(x,[b['validation_loss']]+[v['validation']['eval_loss'] for v in r['replay']],marker='o',label=str(r['source_seed']))
    for axis,label in zip(axes,('Validation accuracy (%)','Validation loss')):
        axis.set(xlabel='Additional round (0 = starting checkpoint)',ylabel=label)
        axis.grid(alpha=.2)
        axis.legend(title='Source seed')
    fig.tight_layout()
    fig.savefig(output/'learning-curves.png',dpi=150)
    plt.close(fig)
    lines += ['',f"Confirmation gate passed: {record['confirmation_gate_passed']}. Promote the refinement: {record['summary']['promote']}.",
        f"Observed mean selected accuracy: {100*selected_acc:.2f}% over {len(runs)} continuation(s); change from the corresponding starting checkpoints: {100*(selected_acc-baseline_acc):+.2f} percentage points.",'',
        'A single-seed pilot is not comparable to the previous three-seed mean as a new confirmed result. Confirmation runs are performed only when the pilot improves loss and gains at least 0.5 percentage points. Promotion requires a mean gain of at least 0.5 points and lower mean loss across all three continuations. These are validation-selected continuations, not independent training from scratch or an equal-compute comparison. Held-out test data remains unused.', '',
        'Communication counts cover these additional rounds only: logical serialized Flower objects including metadata, not captured network traffic. Earlier training, prefix distribution and cache preparation are excluded. Checkpoint hashes, complete rounds, client participation, manifest/source fingerprints and communication totals were audited. No additional experiments follow this batch.', '',
        '![Continuation curves](learning-curves.png)']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    with keep_awake():
        run_batch(args.previous_study,args.output,args.resume)
