"""Run one frozen Top-k+EF pilot against an existing compression study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import statistics
from flower_face.run import launch
from flower_face.study import collect, snapshot, write_json
from flower_face.reproducibility import source_hash


def compatibility(baseline, candidate, *, allow_platform_difference=False):
    keys=('manifest_sha256','initial_model_sha256','versions','platform','reproducibility','selection_rule','fixed_state_policy')
    for key in keys:
        if key == 'platform' and allow_platform_difference:
            continue
        if baseline['metadata'][key]!=candidate['metadata'][key]:
            raise ValueError(f'Baseline compatibility failed: {key}')
    ignored={'compression','output-dir','topk-k','qsgd-levels','llz-p','llz-window'}
    a={k:v for k,v in baseline['config'].items() if k not in ignored}
    b={k:v for k,v in candidate['config'].items() if k not in ignored}
    if a!=b: raise ValueError('Training configurations differ')
    return ({'baseline':baseline['metadata']['platform'],'candidate':candidate['metadata']['platform']}
            if baseline['metadata']['platform']!=candidate['metadata']['platform'] else None)


def report(folder, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for run in runs:
        label=run['config']['compression']
        if label=='qsgd-llz': label+=f"-p{run['config']['llz-p']}"
        rows.append(dict(method=label,selected_round=run['best']['step'],
            accuracy=run['best']['validation_accuracy'],loss=run['best']['validation_loss'],
            final_accuracy=run['final_validation']['eval_acc'],
            upload_bytes=run['train_upload_bytes'],total_bytes=run['total_bytes'],output=run['output']))
    candidate=runs[-1]
    path=Path(candidate['output'])
    timing=[json.loads(x) for x in (path/'codec_timing.jsonl').read_text().splitlines()]
    expected={(r,c) for r in range(1,16) for c in range(4)}
    if len(timing)!=60 or {(x['round'],x['client']) for x in timing}!=expected:
        raise ValueError('Incomplete client timing evidence')
    history=[json.loads(x) for x in (path/'history.jsonl').read_text().splitlines()]
    timing_summary={}
    for key in ('topk_encode_seconds','topk_diagnostic_seconds','decode_seconds'):
        values=[x[key] for x in timing]
        timing_summary[key]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values),
                                first_round=[x[key] for x in timing if x['round']==1])
    timing_summary['round_seconds']=[h['timing']['round_seconds'] for h in history]
    p0=next(row for row in rows if row['method']=='qsgd-llz-p0')
    environment=[compatibility(run,candidate,allow_platform_difference=True) for run in runs[:-1]]
    result=dict(methods=rows,timing=timing_summary,
        environment_differences=[x for x in environment if x],
        strict_environment_match=not any(environment),
        upload_budget_mismatch_percent=100*(rows[-1]['upload_bytes']/p0['upload_bytes']-1),
        accuracy_difference_pp=100*(rows[-1]['accuracy']-p0['accuracy']),
        limitation='One seed; Top-k has feedback while QSGD/LLZ do not. Historical codec timings unavailable. Logical bytes, not network traffic. Any recorded OS difference makes baselines historical context, not fully matched controls.')
    write_json(folder/'summary.json',result)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for row,run in zip(rows,runs):
        curve=[json.loads(x) for x in (Path(run['output'])/'history.jsonl').read_text().splitlines()]
        ax.plot([x['round'] for x in curve],[100*x['validation']['eval_acc'] for x in curve],label=row['method'],linestyle='--' if row['method'].endswith('p0') else '-')
    ax.set(xlabel='Round',ylabel='Validation accuracy (%)',title='Seed 42; historical references (OS build differs)' if any(environment) else 'Seed 42; fixed model and split; feedback differs by method')
    ax.legend();ax.grid(alpha=.2);fig.savefig(folder/'accuracy.png',dpi=150);plt.close(fig)
    lines=['# Top-k with error feedback pilot','',result['limitation'],'',
           '| Method | Selected round | Validation accuracy | Loss | Upload MB | Total MB |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['selected_round']} | {row['accuracy']:.2%} | {row['loss']:.4f} | {row['upload_bytes']/1e6:.3f} | {row['total_bytes']/1e6:.3f} |")
    if any(environment):
        lines += ['', 'Environment difference: '+json.dumps(environment[0])+'. No cross-build causal or bitwise equivalence claim.']
    lines += ['',f"Top-k upload budget mismatch versus p=0: {result['upload_budget_mismatch_percent']:.3f}%.",
              'Selected checkpoint minimizes validation loss. Byte totals cover all rounds. Test split unused.',
              'Timing boundaries and source provenance are in comparison.json and experiment.json. No historical speedup claim.']
    if (folder/'discussion.md').exists():
        lines += ['', '[Post-run literature discussion and timing interpretation](discussion.md).']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--report-only',type=Path,help='Audit/report a completed candidate without training; explicitly label OS differences')
    parser.add_argument('--allow-platform-difference',action='store_true',help='Explicitly permit historical OS-mismatched references; never claim matched environments')
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.report_only:
        folder=args.report_only.resolve(); record=json.loads((folder/'comparison.json').read_text())
        config=record['config']; candidate=collect(folder/'run',config)
        if candidate['metadata']['source_sha256']!=record['source_sha256']:
            raise ValueError('Candidate source differs from saved snapshot')
        runs=[r for r in record['runs'] if r['config']['compression']!='topk-ef']
        for run in runs: compatibility(run,candidate,allow_platform_difference=True)
        runs.append(candidate); record['runs']=runs
        record['summary']=report(folder,runs)
        record['status']='completed' if record['summary']['strict_environment_match'] else 'completed_with_environment_difference'
        record['report_source_sha256']=source_hash(root)
        if 'error' in record: record['original_report_error']=record.pop('error')
        write_json(folder/'comparison.json',record)
        print(json.dumps(record['summary'],indent=2)); return
    if args.baseline is None: parser.error('--baseline is required for training')
    baseline=args.baseline.resolve(); prior=json.loads((baseline/'study.json').read_text())
    if prior['status']!='completed': raise ValueError('Baseline study incomplete')
    if not args.allow_platform_difference and any(r['metadata']['platform']!=platform.platform() for r in prior['runs']):
        raise ValueError('OS differs from historical references; no training launched. Use existing --report-only, or explicitly choose --allow-platform-difference for a new historical-context run.')
    config=dict(prior['protocol']['base_config'])
    if (config['seed'],config['num-server-rounds'],config['num-clients'])!=(42,15,4):
        raise ValueError('Expected agreed seed-42, 15-round, four-client pilot')
    unchanged={}
    for name in ('finetune_task.py','pretrained_task.py','task.py','reproducibility.py'):
        rel='flower_face/'+name
        if (root/rel).read_bytes()!=(baseline/'app'/rel).read_bytes():
            raise ValueError(f'Training implementation changed: {rel}')
        unchanged[rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
    runs=[collect(Path(run['output']).parent,run['config']) for run in prior['runs']]
    folder=(args.output_dir/('topk-ef-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    folder.mkdir(parents=True)
    snapshot(root,folder/'app')
    config.update(compression='topk-ef',**{'topk-k':124000,'evaluate-final-test':False,'output-dir':(folder/'run').as_posix()})
    (folder/'run').mkdir()
    record=dict(status='running',baseline=str(baseline),config=config,unchanged_training_files=unchanged,
                source_sha256=source_hash(folder/'app'),
                compatibility_scope='Same task/model/data/init/hyperparameters; new codec/state/timing and expanded packet accounting. Source hashes intentionally differ.',
                runs=runs)
    write_json(folder/'comparison.json',record)
    print(f'Comparison directory: {folder}',flush=True)
    try:
        code=launch(config,root=root,app_dir=folder/'app',log_path=folder/'terminal.log')
        if code: raise RuntimeError(f'Flower failed ({code}); see terminal.log')
        candidate=collect(folder/'run',config)
        if candidate['metadata']['source_sha256']!=record['source_sha256']:
            raise ValueError('Run source differs from snapshot')
        for run in runs: compatibility(run,candidate,allow_platform_difference=args.allow_platform_difference)
        runs.append(candidate)
        record['summary']=report(folder,runs)
        record['status']='completed' if record['summary']['strict_environment_match'] else 'completed_with_environment_difference'
    except BaseException as error:
        record.update(status='failed',error=str(error));write_json(folder/'comparison.json',record);raise
    write_json(folder/'comparison.json',record)
    print(json.dumps(record['summary'],indent=2),flush=True)


if __name__=='__main__': main()
