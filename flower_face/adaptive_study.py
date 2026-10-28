"""Calibrate on eight saved train updates, freeze, then run five matched methods.

No extra training seeds, dataset changes, feedback, optimizer tuning or test
evaluation. Resume reuses completed runs from this frozen app only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
from time import perf_counter

from flwr.app import Message, RecordDict
import numpy as np
import torch

from compression.adaptive_llz import admissible
from federated_compression.adaptive_llz import encode_reply
from flower_face import finetune_task as task
from flower_face.check_llz import compare
from flower_face.communication import measure_message
from flower_face.reproducibility import model_hash, source_hash
from flower_face.run import default_config, launch
from flower_face.study import collect, snapshot, write_json
from flower_face.task import seed_everything

METHODS = ('qsgd','qsgd-llz-p0','qsgd-llz-p1','adaptive-tight','adaptive-loose')
CALIBRATION = Path('outputs/ae-calibration-20261005T122126176771Z')
REFERENCE = Path('outputs/compression-controls-20261012T073109367046Z/comparison.json')


def choose_budgets(rows):
    """A priori rule: lower quartile of p1, upper quartile of p2; two sig figs."""
    p1 = [r['relative_errors'][1] for r in rows]
    p2 = [r['relative_errors'][2] for r in rows]
    positive = [v for v in p1+p2 if v > 0 and math.isfinite(v)]
    if not positive or any(not math.isfinite(v) or v < 0 for v in p1+p2):
        raise ValueError('Calibration cannot identify positive distortion budgets')
    def rounded(value, up):
        scale=10.**(math.floor(math.log10(value))-1)
        return float(format((math.ceil(value/scale) if up else math.floor(value/scale))*scale,'.12g'))
    tight_raw=float(np.quantile(p1,.25))
    if tight_raw == 0:
        tight_raw=min(positive)/2
    tight=rounded(tight_raw,False)
    loose_raw=max(float(np.quantile(p2,.75)),tight*1.1)
    loose=rounded(loose_raw,True)
    if not 0 < tight < loose:
        raise ValueError('Expected two distinct positive calibration budgets')
    return dict(tight=tight,loose=loose,
        rule='tight=floor to 2 significant digits of p1 relative-error Q25; loose=ceil to 2 significant digits of max(p2 Q75, 1.1*tight). If p1 Q25 is zero, use half the smallest positive candidate error before rounding.',
        raw_p1_q25=float(np.quantile(p1,.25)),raw_p2_q75=float(np.quantile(p2,.75)))


def calibrate_one(job):
    torch.set_num_threads(1)
    folder, client, trajectory, snap, config = job
    file=Path(folder)/snap['full_delta']['path']
    if task.file_hash(file)!=snap['full_delta']['sha256'] or file.stat().st_size!=snap['full_delta']['bytes']:
        raise ValueError('Saved update checksum mismatch')
    with np.load(file,allow_pickle=False) as data:
        delta={name:torch.from_numpy(data[name].copy()) for name in data.files}
    if (sorted(delta)!=trajectory['names'] or model_hash(delta)!=snap['delta_sha256']
            or any(v.dtype!=torch.float32 or list(v.shape)!=trajectory['shapes'][k] for k,v in delta.items())):
        raise ValueError('Saved update layout or delta hash mismatch')
    instruction=Message(RecordDict(),dst_node_id=client+1,message_type='train')
    started=perf_counter()
    reply=encode_reply(delta,{k:torch.zeros_like(v) for k,v in delta.items()},
        levels=127,seed=42,server_round=1,client_id=client,budget=1.e100,window=128,
        metrics={'num-examples':750,'train_loss':float(trajectory['training_loss'])},
        instruction=instruction,measure=measure_message)
    stats=dict(reply.content['metrics'])
    norm=stats['qsgd_input_squared_norm']
    errors=stats['adaptive_candidate_squared_error']
    if norm<=0 or errors[0]!=0:
        raise ValueError('Expected nonzero calibration update and exact p0')
    return dict(id=snap['id'],client=client,step=snap['step'],source_file=str(file.resolve()),
        source_sha256=snap['full_delta']['sha256'],delta_sha256=snap['delta_sha256'],
        original_trajectory_seed=trajectory['seed'],codec_round=1,
        relative_errors=[v/norm for v in errors],elapsed_seconds=perf_counter()-started,
        candidate_stats=stats)


def compatible_collection(collection, config):
    if collection['status']!='completed' or collection['audit']['status']!='passed':
        raise ValueError('Calibration requires the completed audited saved-update collection')
    protocol=collection['protocol']
    ignore={'output-dir','compression','llz-budget','topk-k','llz-p','qsgd-levels','llz-window','evaluate-final-test'}
    previous=protocol['config']
    if ({k:v for k,v in config.items() if k not in ignore}
            != {k:v for k,v in previous.items() if k not in ignore}):
        raise ValueError('Saved update training configuration differs')
    if task.file_hash(config['manifest'])!=protocol['manifest_sha256']:
        raise ValueError('Calibration manifest differs')
    seed_everything(config['seed'])
    model=task.build_model(config,config['num-classes'])
    if model_hash(model.state_dict())!=protocol['initial_model_sha256']:
        raise ValueError('Saved updates use different initialization')
    fixed=task.fixed_state_names(model)
    shapes={k:list(v.shape) for k,v in model.state_dict().items() if k not in fixed}
    clients=collection['clients']
    if {c['client'] for c in clients}!=set(range(4)) or len(clients)!=4:
        raise ValueError('Expected four saved clients')
    for client in clients:
        trajectories=[t for t in client['trajectories'] if t['role']=='validation']
        if len(trajectories)!=1 or trajectories[0]['shapes']!=shapes or set(trajectories[0]['fixed_buffers'])!=fixed:
            raise ValueError('Saved update tensor/fixed-buffer policy differs')
    return protocol


def prepare(root):
    previous=json.loads((root/REFERENCE).read_text())
    config=default_config(root)
    config.update(previous['config'])
    config.update({'compression':'qsgd','llz-p':0,'llz-budget':0.,'evaluate-final-test':False})
    if (config['seed'],config['num-server-rounds'],config['num-clients'],config['qsgd-levels'],config['llz-window'])!=(42,15,4,127,128):
        raise ValueError('Expected the agreed 15-round seed-42 four-client settings')
    collection=json.loads((root/CALIBRATION/'collection.json').read_text())
    saved_protocol=compatible_collection(collection,config)
    folder=root/'outputs'/('adaptive-study-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    snapshot(root,folder/'app')
    record=dict(status='calibrating',config=config,source_sha256=source_hash(folder/'app'),
        manifest_sha256=saved_protocol['manifest_sha256'],initial_model_sha256=saved_protocol['initial_model_sha256'],
        environment=dict(platform=platform.platform(),python=platform.python_version(),
            versions={k:version(k) for k in ('flwr','torch','torchvision','numpy','ray')}),
        calibration_source=str((root/CALIBRATION).resolve()),calibration=[],runs=[],attempts=[],launch_seconds={},methods=list(METHODS))
    write_json(folder/'study.json',record)
    jobs=[]
    for client in sorted(collection['clients'],key=lambda c:c['client']):
        trajectory=next(t for t in client['trajectories'] if t['role']=='validation')
        snaps=trajectory['snapshots']
        for snap in (snaps[(len(snaps)-1)//2],snaps[-1]):
            jobs.append((str(root/CALIBRATION),client['client'],trajectory,snap,config))
    print(f'Study directory: {folder}',flush=True)
    try:
        # Four existing client updates at a time; no local training is performed.
        os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(calibrate_one,job) for job in jobs]
            for future in as_completed(futures):
                row=future.result();record['calibration'].append(row)
                write_json(folder/'study.json',record)
                print(f"Calibrated {row['id']}: p1/p2 error={row['relative_errors'][1]:.6g}/{row['relative_errors'][2]:.6g}",flush=True)
        record['calibration'].sort(key=lambda r:(r['client'],r['step']))
        record['budgets']=choose_budgets(record['calibration'])
        if source_hash(root)!=record['source_sha256']:
            raise ValueError('Source changed during calibration; no training launched')
        protocol=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            config=config,methods=list(METHODS),budgets=record['budgets'],
            source_sha256=record['source_sha256'],manifest_sha256=record['manifest_sha256'],
            calibration_scope='8 training-image-only saved local deltas (4 clients x midpoint/final); independent reset trajectories, not global FL rounds; no identity-validation/test selection',
            fallback='p0 always feasible under additional-error constraint; codec failures stop the run',
            selection='one p per client update; actual full reply bytes; lower p breaks ties',
            calibration_framing='Full adaptive reply with surrogate small routing IDs; no physical upload; final runtime bytes logged separately',
            run_order=list(METHODS),extra_experiments='none; no added seeds, optimizers, datasets or feedback',
            audit='exact QSGD/p0 all-round/checkpoint replay; every adaptive candidate feasibility/optimality and serialized-byte reconciliation; complete client participation; held-out test unused')
        write_json(folder/'protocol.json',protocol)
        record['protocol_sha256']=task.file_hash(folder/'protocol.json')
        record['status']='prepared'
        write_json(folder/'study.json',record)
        print(f"Frozen budgets: tight={record['budgets']['tight']}, loose={record['budgets']['loose']}",flush=True)
    except BaseException as error:
        record.update(status='failed',error=str(error));write_json(folder/'study.json',record);raise
    return folder


def run_config(record,label,output):
    config=dict(record['config'])
    config.update({'output-dir':str(output).replace('\\','/'),'llz-p':0,'llz-budget':0.})
    if label=='qsgd': config['compression']='qsgd'
    elif label in ('qsgd-llz-p0','qsgd-llz-p1'):
        config.update({'compression':'qsgd-llz','llz-p':int(label[-1])})
    elif label in ('adaptive-tight','adaptive-loose'):
        config.update({'compression':'qsgd-llz-adaptive','llz-budget':record['budgets'][label.removeprefix('adaptive-')]})
    else: raise ValueError('Unexpected method')
    return config


def audit(record):
    runs=record['runs']
    if len(runs)!=5 or [r['label'] for r in runs]!=list(METHODS):
        raise ValueError('Expected exactly the five frozen methods')
    reference=runs[0]
    exact=compare(reference,runs[1])
    methods={}
    for run in runs:
        for key in ('source_sha256','manifest_sha256','initial_model_sha256','versions','platform','python',
                    'reproducibility','fixed_state_policy','selection_rule','communication_measurement'):
            if run['metadata'][key]!=reference['metadata'][key]:
                raise ValueError(f'Matched-study metadata differs: {key}')
        ignored={'compression','llz-p','llz-budget','output-dir'}
        if ({k:v for k,v in run['config'].items() if k not in ignored}
                != {k:v for k,v in reference['config'].items() if k not in ignored}):
            raise ValueError('Matched-study training configuration differs')
        name=run['label']
        if run['config']!=run_config(record,name,run['config']['output-dir']):
            raise ValueError('Run differs from frozen compression policy')
        row=dict(output=run['output'],best=run['best'],final_validation=run['final_validation'],
            upload_bytes=run['train_upload_bytes'],total_bytes=run['total_bytes'],
            selected_accuracy_difference_vs_qsgd_pp=100*(run['best']['validation_accuracy']-reference['best']['validation_accuracy']),
            upload_saving_vs_qsgd_percent=100*(1-run['train_upload_bytes']/reference['train_upload_bytes']),
            total_saving_vs_qsgd_percent=100*(1-run['total_bytes']/reference['total_bytes']))
        if name.startswith('adaptive-'):
            folder=Path(run['output'])
            rows=[json.loads(s) for s in (folder/'adaptive_selection.jsonl').read_text().splitlines()]
            ledger=[json.loads(s) for s in (folder/'communication_messages.jsonl').read_text().splitlines()]
            uploads={(r['round'],r['client_partition_id']):r for r in ledger if r['phase']=='train' and r['direction']=='uplink'}
            expected={(r,c) for r in range(1,16) for c in range(4)}
            if len(rows)!=60 or {(r['round'],r['client']) for r in rows}!=expected:
                raise ValueError('Missing adaptive client/round diagnostics')
            for r in rows:
                norm=r['qsgd_input_squared_norm'];budget=run['config']['llz-budget'];p=r['adaptive_selected_p']
                errors=r['adaptive_candidate_squared_error'];costs=r['adaptive_candidate_message_bytes']
                feasible=[i for i in (0,1,2) if admissible(errors[i],norm,budget)]
                actual=uploads[r['round'],r['client']]
                if (errors[0]!=0 or not feasible or p!=min(feasible,key=lambda i:(costs[i],i))
                        or not admissible(r['llz_reconstruction_squared_error'],norm,budget)
                        or r['adaptive_budget']!=budget or r['observed_message_bytes']!=actual['serialized_object_bytes']
                        or r['adaptive_selected_message_bytes']!=r['observed_message_bytes']
                        or r['adaptive_candidate_payload_bytes'][p]!=actual['codec_payload_bytes']):
                    raise ValueError('Adaptive policy, distortion or byte-accounting audit failed')
            row.update(budget=run['config']['llz-budget'],
                selected_p_counts={str(p):sum(r['adaptive_selected_p']==p for r in rows) for p in (0,1,2)},
                max_extra_relative_squared_error=max(r['llz_reconstruction_squared_error']/r['qsgd_input_squared_norm']
                    if r['qsgd_input_squared_norm'] else 0. for r in rows),
                selection_seconds_sum=sum(r['adaptive_selection_seconds'] for r in rows),
                decode_seconds_sum=sum(r['decode_seconds'] for r in rows))
        methods[name]=row
    return dict(exact_p0_replay=exact,methods=methods,
        scope='One seed, 15 rounds, fixed DigiFace model/data/training, no feedback; selected by validation loss; held-out test unused. Logical serialized object bytes, not network traffic. No significance or convergence claim.')


def train(root,folder):
    record=json.loads((folder/'study.json').read_text());stage=folder/'app'
    if (not folder.is_relative_to(root/'outputs') or source_hash(stage)!=record['source_sha256']
            or task.file_hash(folder/'protocol.json')!=record['protocol_sha256']
            or task.file_hash(record['config']['manifest'])!=record['manifest_sha256']):
        raise ValueError('Frozen source/protocol/manifest changed or output outside outputs/')
    current=dict(platform=platform.platform(),python=platform.python_version(),
                 versions={k:version(k) for k in record['environment']['versions']})
    if current!=record['environment']: raise ValueError('Experiment environment changed')
    if record['status']=='completed':
        print(f'Completed batch reused: {folder}',flush=True);return
    for saved in record['runs']:
        actual=collect(Path(saved['output']).parent,saved['config'])
        if actual!={k:v for k,v in saved.items() if k!='label'}:
            raise ValueError('Completed saved run changed; no retraining launched')
    record['status']='running';record.pop('error',None);write_json(folder/'study.json',record)
    print(f'Frozen comparison: {folder}',flush=True)
    try:
        for label in METHODS:
            if any(r['label']==label for r in record['runs']): continue
            number=1+sum(a['label']==label for a in record['attempts'])
            run_root=folder/f'{label}-attempt-{number}';run_root.mkdir()
            config=run_config(record,label,run_root.as_posix())
            attempt=dict(label=label,output=str(run_root),status='running')
            record['attempts'].append(attempt);write_json(folder/'study.json',record)
            print(f'Starting {label}: seed42, 15 rounds',flush=True)
            def progress(line):
                if 'serialized objects:' in line or 'Best checkpoint:' in line:
                    print(f'[{label}] {line}',flush=True)
            started=perf_counter()
            code=launch(config,root=root,app_dir=stage,log_path=run_root/'terminal.log',progress=progress)
            elapsed=perf_counter()-started
            if code: raise RuntimeError(f'Flower failed ({code}): {run_root / "terminal.log"}')
            result=collect(run_root,config)
            if any(result['metadata'][k]!=record[k] for k in ('source_sha256','manifest_sha256','initial_model_sha256')):
                raise ValueError('Run source/data/initialization differs from frozen protocol')
            result['label']=label;record['runs'].append(result)
            record['launch_seconds'][label]=elapsed;attempt['status']='completed'
            write_json(folder/'study.json',record)
            print(f"Completed {label}: selected accuracy={result['best']['validation_accuracy']:.2%}; uploads={result['train_upload_bytes']:,}; seconds={elapsed:.1f}",flush=True)
            if label=='qsgd-llz-p0':
                compare(record['runs'][0],result)
                print('Exact p0 replay gate passed.',flush=True)
        record['summary']=audit(record);write_json(folder/'summary.json',record['summary'])
        record['status']='completed';write_json(folder/'study.json',record)
        print(f'Completed audited five-method adaptive pilot: {folder}',flush=True)
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        write_json(folder/'study.json',record);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    folder=args.resume.resolve() if args.resume else prepare(root)
    if not args.prepare_only: train(root,folder)


if __name__=='__main__': main()
