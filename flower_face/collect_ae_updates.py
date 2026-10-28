"""Collect one bounded training-only calibration batch for a learned compressor.

Four clients, two reset one-epoch trajectories each, at most ten snapshots per
trajectory. Fitting retains sampled blocks; validation retains full model deltas.
No Flower communication, identity-validation evaluation, or compressor fitting.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
from time import perf_counter

import numpy as np
import torch
from compression.block_data import BLOCK_SIZE,padded_blocks,normalize_blocks,stratified_counts,snapshot_steps
from flower_face import finetune_task as task
from flower_face.task import seed_everything
from flower_face.task_api import verify_fixed_state
from flower_face.reproducibility import model_hash,source_hash
from flower_face.study import snapshot,write_json


def collection_source_hash(root):
    relevant=('flower_face/collect_ae_updates.py','compression/block_data.py',
              'flower_face/finetune_task.py','flower_face/pretrained_task.py',
              'flower_face/task.py','flower_face/reproducibility.py','flower_face/task_api.py')
    entries={name:hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() for name in relevant}
    return hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()


def file_entry(path,base):
    return dict(path=path.relative_to(base).as_posix(),bytes=path.stat().st_size,
                sha256=task.file_hash(path))


def collect_client(client,config,output,expected_initial):
    started=perf_counter(); output=Path(output)
    directory=output/f'client-{client}'; directory.mkdir()
    manifest=task.read_manifest(config['manifest'],config['num-classes'],config['num-clients'])
    rows=[x for x in manifest['examples'] if x['split']=='train' and x['client_id']==client]
    summary=dict(client=client,source_image_split='train',training_examples=len(rows),
        training_filenames_sha256=hashlib.sha256(json.dumps(sorted(x['filename'] for x in rows)).encode()).hexdigest(),
        trajectories=[])
    for role,trajectory,budget in [('fit',0,32768),('validation',1,8192)]:
        preparation_started=perf_counter()
        seed_everything(config['seed'])
        model=task.build_model(config,config['num-classes'])
        if model_hash(model.state_dict())!=expected_initial:
            raise ValueError('Calibration initialization differs from reference')
        fixed=task.fixed_state_names(model)
        base={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k not in fixed}
        fixed_reference={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in fixed}
        names=sorted(base)
        if any(v.dtype!=torch.float32 for v in base.values()):
            raise ValueError('Calibration requires float32 trainable state')
        trajectory_seed=config['seed']+1000*client+100000*trajectory
        seed_everything(trajectory_seed)
        loader=task.load_data(config,client,'train',trajectory_seed)
        endpoints=snapshot_steps(len(loader))
        population=[math.ceil(base[name].numel()/BLOCK_SIZE) for _ in endpoints for name in names]
        counts=stratified_counts(population,budget).reshape(len(endpoints),len(names))
        total=int(counts.sum())
        data={
            'blocks':np.empty((total,BLOCK_SIZE),np.float32),
            'scales':np.empty(total,np.float32),
            'population_weights':np.empty(total,np.float64),
            'tensor_id':np.empty(total,np.int32),
            'block_index':np.empty(total,np.int32),
            'snapshot_step':np.empty(total,np.int32),
            'valid_length':np.empty(total,np.int32),
        }
        trajectory_dir=directory/role;trajectory_dir.mkdir()
        snapshots=[];cursor=0; snapshot_seconds=0.0
        def capture(step,current):
            nonlocal cursor,snapshot_seconds
            if step not in endpoints: return
            tick=perf_counter(); verify_fixed_state(current.state_dict(),fixed_reference)
            delta={name:(current.state_dict()[name].detach().cpu()-base[name]).numpy() for name in names}
            if any(not np.isfinite(x).all() for x in delta.values()): raise ValueError('Nonfinite calibration delta')
            entry=dict(id=f'client-{client}:{role}:step-{step}',step=step,
                       delta_sha256=model_hash({k:torch.from_numpy(v) for k,v in delta.items()}),strata=[])
            if role=='validation':
                file=trajectory_dir/f'delta-{step:03d}.npz'
                np.savez(file,**delta);entry['full_delta']=file_entry(file,output)
            si=endpoints.index(step)
            for ti,name in enumerate(names):
                blocks=padded_blocks(delta[name]);n=blocks.shape[0];k=int(counts[si,ti])
                sampling_seed=[config['seed'],client,trajectory,step,ti,0x4145]
                rng=np.random.default_rng(np.random.SeedSequence(sampling_seed))
                chosen=np.sort(rng.choice(n,size=k,replace=False))
                normalized,scales=normalize_blocks(blocks[chosen])
                sl=slice(cursor,cursor+k)
                data['blocks'][sl]=normalized;data['scales'][sl]=scales
                data['population_weights'][sl]=n/k
                data['tensor_id'][sl]=ti;data['block_index'][sl]=chosen
                data['snapshot_step'][sl]=step
                data['valid_length'][sl]=np.minimum(BLOCK_SIZE,delta[name].size-chosen*BLOCK_SIZE)
                entry['strata'].append(dict(tensor=name,tensor_id=ti,population_blocks=n,
                                            sampled_blocks=k,weight=n/k,sampling_seed=sampling_seed))
                cursor+=k
            snapshots.append(entry)
            snapshot_seconds+=perf_counter()-tick
        preparation_seconds=perf_counter()-preparation_started
        training_started=perf_counter()
        loss=task.train(model,loader,1,config['learning-rate'],
                        weight_decay=config.get('weight-decay',0.0),on_batch=capture)
        training_including_collection=perf_counter()-training_started
        verify_fixed_state(model.state_dict(),fixed_reference)
        if cursor!=total or [x['step'] for x in snapshots]!=endpoints:
            raise ValueError('Incomplete sampled snapshot coverage')
        tick=perf_counter();sample_file=trajectory_dir/'sampled-blocks.npz'
        np.savez(sample_file,**data);sample_entry=file_entry(sample_file,output)
        sample_write_seconds=perf_counter()-tick
        record=dict(role=role,trajectory=trajectory,seed=trajectory_seed,epochs=1,batches=len(loader),
            initial_model_sha256=expected_initial,names=names,shapes={k:list(v.shape) for k,v in base.items()},
            fixed_buffers=sorted(fixed),population_blocks=sum(population),sampled_blocks=total,
            training_loss=loss,snapshots=snapshots,sampled_file=sample_entry,
            preparation_seconds=preparation_seconds,
            training_including_collection_seconds=training_including_collection,
            snapshot_processing_seconds=snapshot_seconds,
            local_training_excluding_snapshot_callbacks_seconds=training_including_collection-snapshot_seconds,
            sampled_file_write_seconds=sample_write_seconds,
            residual_policy='none; cumulative delta from common initial model',
            calibration_protocol='independent local trajectory, not a federated training run')
        write_json(trajectory_dir/'trajectory.json',record)
        summary['trajectories'].append(record)
        print(f'Client {client} {role}: {len(snapshots)} snapshots, {total} sampled blocks',flush=True)
    summary['elapsed_seconds']=perf_counter()-started
    write_json(directory/'client.json',summary)
    return summary


def audit(folder,record):
    """One completed-batch check of hashes, shape/sampling lineage and isolation."""
    folder=Path(folder);ids=set();files=0;bytes_total=0
    if {c['client'] for c in record['clients']}!=set(range(4)): raise ValueError('Incomplete client batch')
    for client in record['clients']:
        if [t['role'] for t in client['trajectories']]!=['fit','validation']:
            raise ValueError('Missing fitting or validation trajectory')
        for trajectory in client['trajectories']:
            entries=[trajectory['sampled_file']]+[s['full_delta'] for s in trajectory['snapshots'] if 'full_delta' in s]
            for entry in entries:
                file=folder/entry['path']
                if file.stat().st_size!=entry['bytes'] or task.file_hash(file)!=entry['sha256']:
                    raise ValueError('Collected file checksum mismatch')
                files+=1;bytes_total+=entry['bytes']
            for snap in trajectory['snapshots']:
                if snap['id'] in ids: raise ValueError('Overlapping update IDs')
                ids.add(snap['id'])
                if trajectory['role']=='validation':
                    with np.load(folder/snap['full_delta']['path'],allow_pickle=False) as delta:
                        if sorted(delta.files)!=trajectory['names']: raise ValueError('Delta names differ')
                        for name in delta.files:
                            x=delta[name]
                            if x.dtype!=np.float32 or list(x.shape)!=trajectory['shapes'][name] or not np.isfinite(x).all():
                                raise ValueError('Invalid full validation delta')
                        if model_hash({k:torch.from_numpy(delta[k]) for k in delta.files})!=snap['delta_sha256']:
                            raise ValueError('Delta hash mismatch')
            with np.load(folder/trajectory['sampled_file']['path'],allow_pickle=False) as samples:
                blocks=samples['blocks'];n=trajectory['sampled_blocks']
                if blocks.shape!=(n,BLOCK_SIZE) or blocks.dtype!=np.float32 or not np.isfinite(blocks).all():
                    raise ValueError('Invalid sampled blocks')
                if samples['scales'].shape!=(n,) or not np.isfinite(samples['scales']).all() or np.any(samples['scales']<0):
                    raise ValueError('Invalid sampled norms')
                if np.any(samples['population_weights']<=0): raise ValueError('Invalid sampling weights')
                triples=np.stack([samples['snapshot_step'],samples['tensor_id'],samples['block_index']],axis=1)
                if np.unique(triples,axis=0).shape[0]!=n: raise ValueError('Duplicate sampled blocks')
                for snap in trajectory['snapshots']:
                    for stratum in snap['strata']:
                        mask=(samples['snapshot_step']==snap['step'])&(samples['tensor_id']==stratum['tensor_id'])
                        if int(mask.sum())!=stratum['sampled_blocks'] or not np.all(samples['population_weights'][mask]==stratum['weight']):
                            raise ValueError('Sampling strata do not reconcile')
                zeros=samples['scales']==0
                if np.any(blocks[zeros]!=0): raise ValueError('Zero-norm block not zero')
                for i,length in enumerate(samples['valid_length']):
                    if not 1<=length<=BLOCK_SIZE or np.any(blocks[i,int(length):]!=0):
                        raise ValueError('Invalid padding')
    return dict(status='passed',unique_snapshot_ids=len(ids),data_files=files,data_bytes=bytes_total,
                identity_validation_or_test_used=False,
                split_scope='Disjoint trajectories, shared training-image population; not unseen-client generalization')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('outputs/topk-ef-20260930T113916818351Z/comparison.json'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=4: parser.error('workers must be between one and four')
    root=Path(__file__).resolve().parents[1]
    prior=json.loads(args.reference.read_text());ref=prior['runs'][-1]
    config=dict(ref['config']);config['compression']='none';config['evaluate-final-test']=False
    if (config['seed'],config['num-clients'],config['num-classes'],config['local-epochs'])!=(42,4,100,1):
        raise ValueError('Expected agreed four-client seed-42 fine-tuning reference')
    if config['task-module']!='flower_face.finetune_task': raise ValueError('Collector needs fine-tuning task')
    expected=ref['metadata']['initial_model_sha256']
    protocol=dict(config={k:v for k,v in config.items() if k!='output-dir'},initial_model_sha256=expected,
        manifest_sha256=task.file_hash(Path(config['manifest'])),collection_source_sha256=collection_source_hash(root),
        clients=4,trajectories_per_client=2,epochs_per_trajectory=1,snapshot_limit=10,
        fitting_block_cap=32768,validation_block_cap=8192,block_size=BLOCK_SIZE,
        trajectory_seed='seed + 1000*client + 100000*trajectory',
        sampling_seed='SeedSequence([seed, client, trajectory, step, sorted_tensor_id, 0x4145])',
        source_split='train only; role validation means update reconstruction, not identity validation')
    for existing in sorted(args.output_dir.glob('ae-calibration-*'),reverse=True):
        saved=existing/'collection.json'
        if saved.exists():
            record=json.loads(saved.read_text())
            if record.get('status')=='completed' and record.get('protocol')==protocol:
                audit(existing,record);print(f'Reusing verified collection: {existing.resolve()}');return
    required=2_000_000_000
    if shutil.disk_usage(root).free<required: raise ValueError('Need at least 2 GB free for bounded calibration batch')
    folder=(args.output_dir/('ae-calibration-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    folder.mkdir(parents=True,exist_ok=False)
    snapshot(root,folder/'app')
    shutil.copy2(config['manifest'],folder/'manifest.json')
    started=perf_counter()
    record=dict(status='running',protocol=protocol,source_sha256=source_hash(root),clients=[],workers=args.workers,
        environment=dict(platform=platform.platform(),python=platform.python_version(),versions={name:version(name) for name in ('flwr','torch','torchvision','numpy','ray')}),
        measurement='CPU local calibration; includes collection overhead; no Flower/network bytes measured',
        reference=str(args.reference.resolve()))
    write_json(folder/'collection.json',record)
    print(f'Collection directory: {folder}',flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures={executor.submit(collect_client,c,config,str(folder),expected):c for c in range(4)}
            for future in as_completed(futures):
                record['clients'].append(future.result());record['clients'].sort(key=lambda x:x['client'])
                write_json(folder/'collection.json',record)
        record['elapsed_seconds']=perf_counter()-started
        if collection_source_hash(root)!=protocol['collection_source_sha256']:
            raise ValueError('Collection implementation changed during the batch')
        record['audit']=audit(folder,record)
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=str(error));write_json(folder/'collection.json',record);raise
    write_json(folder/'collection.json',record)
    print(json.dumps(record['audit'],indent=2),flush=True)
    print(f'Collection completed in {record["elapsed_seconds"]:.1f} seconds',flush=True)


if __name__=='__main__': main()
