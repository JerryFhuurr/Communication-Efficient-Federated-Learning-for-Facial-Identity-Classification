"""Bounded adaptive integration check: three two-round runs, four clients.

Reuses data/synthetic; this is a runtime correctness check, not a DigiFace
accuracy comparison or a calibration of thesis distortion budgets.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from flower_face.reproducibility import source_hash
from flower_face.run import default_config, launch
from flower_face.study import collect, snapshot, write_json


def audit(runs):
    """Audit completed runs once; require exact zero-budget learning replay."""
    qsgd, zero, positive = runs
    for run in runs:
        for key in ('manifest_sha256','source_sha256','versions','platform',
                    'reproducibility','initial_model_sha256','selection_rule','communication_measurement'):
            if run['metadata'][key] != qsgd['metadata'][key]:
                raise RuntimeError(f'Integration run mismatch: {key}')
        excluded = {'compression','output-dir','llz-budget'}
        if ({k:v for k,v in run['config'].items() if k not in excluded}
                != {k:v for k,v in qsgd['config'].items() if k not in excluded}):
            raise RuntimeError('Integration training settings differ')
    def learning(run):
        # Codec diagnostics/timings intentionally differ; learning must not.
        return [dict(round=row['round'], model_sha256=row['model_sha256'],
            train={k:v for k,v in row['train'].items() if not k.startswith(('qsgd_','llz_','adaptive_'))},
            validation=row['validation']) for row in run['replay']]
    if learning(qsgd) != learning(zero) or qsgd['best'] != zero['best']:
        raise RuntimeError('Zero-budget QSGD learning/checkpoint replay failed')
    diagnostics = {}
    for label, run in (('zero',zero), ('positive',positive)):
        folder = Path(run['output'])
        rows = [json.loads(line) for line in (folder/'adaptive_selection.jsonl').read_text().splitlines()]
        ledger = [json.loads(line) for line in (folder/'communication_messages.jsonl').read_text().splitlines()]
        uploads = {(r['round'],r['client_partition_id']):r for r in ledger
                   if r['phase']=='train' and r['direction']=='uplink'}
        expected = {(r,c) for r in range(1,run['config']['num-server-rounds']+1)
                    for c in range(run['config']['num-clients'])}
        if len(rows)!=len(expected) or {(r['round'],r['client']) for r in rows}!=expected:
            raise RuntimeError('Incomplete adaptive client diagnostics')
        for row in rows:
            upload = uploads[row['round'],row['client']]
            norm, error = row['qsgd_input_squared_norm'], row['llz_reconstruction_squared_error']
            if ((error!=0 if norm==0 else error/norm>run['config']['llz-budget'])
                    or row['observed_message_bytes'] != upload['serialized_object_bytes']
                    or row['adaptive_selected_message_bytes'] != row['observed_message_bytes']):
                raise RuntimeError('Adaptive budget or byte-accounting check failed')
        diagnostics[label] = dict(
            selections={str(p):sum(r['adaptive_selected_p']==p for r in rows) for p in (0,1,2)},
            max_extra_relative_squared_error=max(r['llz_reconstruction_squared_error']/r['qsgd_input_squared_norm']
                if r['qsgd_input_squared_norm'] else 0. for r in rows),
            selection_seconds_sum=sum(r['adaptive_selection_seconds'] for r in rows),
            train_upload_bytes=run['train_upload_bytes'])
    if not any(diagnostics['positive']['selections'][str(p)] for p in (1,2)):
        raise RuntimeError('Positive-budget integration did not exercise a positive-p upload')
    return dict(exact_zero_budget_replay=True, rounds=len(qsgd['replay']), diagnostics=diagnostics,
        test_evaluated=False,
        scope='Four-client synthetic runtime check; not facial accuracy evidence or budget calibration.',
        measurement='Logical serialized Flower Message object graph, including diagnostics; not network traffic.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = default_config(root)
    base.update({'task-module':'flower_face.synthetic_task',
        'manifest':(root/'data/synthetic/manifest.json').as_posix(),
        'num-classes':3,'num-clients':4,'num-server-rounds':2,'seed':42,
        'local-epochs':1,'learning-rate':.1,'weight-decay':0.,'batch-size':16,
        'augmentation':'none','initial-checkpoint':'','initial-checkpoint-sha256':'',
        'qsgd-levels':127,'llz-p':0,'llz-window':128,'evaluate-final-test':False})
    output = args.output_dir.resolve()/('adaptive-check-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    stage = output/'app'
    snapshot(root,stage)
    record = dict(status='running',source_sha256=source_hash(stage),runs=[],
                  plan=[['qsgd',0.],['adaptive-zero',0.],['adaptive-positive',1.]])
    write_json(output/'check.json',record)
    print(f'Integration directory: {output}',flush=True)
    try:
        for label,budget in record['plan']:
            run_root=output/label
            run_root.mkdir()
            config=dict(base,compression='qsgd' if label=='qsgd' else 'qsgd-llz-adaptive')
            config.update({'llz-budget':budget,'output-dir':run_root.as_posix()})
            print(f'Starting {label}: two rounds, four synthetic clients',flush=True)
            if launch(config,root=root,app_dir=stage,log_path=run_root/'terminal.log') != 0:
                raise RuntimeError(f'Flower failed: inspect {run_root / "terminal.log"}')
            record['runs'].append(collect(run_root,config))
            write_json(output/'check.json',record)
        record['audit']=audit(record['runs'])
        record['status']='completed'
        write_json(output/'check.json',record)
        print('Passed: exact zero-budget replay, positive-p uploads, and byte/budget accounting.',flush=True)
    except Exception as error:
        record.update(status='failed',error=str(error))
        write_json(output/'check.json',record)
        raise


if __name__=='__main__':
    main()
