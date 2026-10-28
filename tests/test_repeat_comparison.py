from copy import deepcopy
from pathlib import Path
import pytest
from flower_face.repeat_comparison import plan, check_run, audit_seed
from test_complete_comparison import rows


def base():
    return dict(seed=42, **{'num-server-rounds':15, 'num-clients':4,
        'qsgd-levels':127, 'llz-window':128, 'topk-k':124000,
        'evaluate-final-test':False, 'learning-rate':0.003})


def test_plan_is_exactly_requested_matrix_and_keeps_training_settings():
    configs=plan(base(), Path('outputs/repeats'))
    assert len(configs)==10
    assert [r['seed'] for r in configs]==[43]*5+[44]*5
    assert len({r['output-dir'] for r in configs})==10
    assert [(r['compression'],r['llz-p']) for r in configs[:5]]==[
        ('none',0),('qsgd',0),('qsgd-llz',0),('qsgd-llz',1),('topk-ef',0)]
    for r in configs:
        for k in ('num-server-rounds','num-clients','learning-rate','qsgd-levels','llz-window','topk-k','evaluate-final-test'):
            assert r[k]==base()[k]


def test_plan_rejects_drift_and_checks_frozen_run_provenance():
    changed=base();changed['num-server-rounds']=16
    with pytest.raises(ValueError): plan(changed,Path('outputs/repeats'))
    a=rows()[0]
    # Real experiment metadata includes all fields; add fixture-only omissions.
    for k in ('python','communication_measurement'): a['metadata'].setdefault(k,'same')
    b=deepcopy(a);b['config']['seed']=43;b['metadata']['initial_model_sha256']='new-seed'
    check_run(a,b,b['config'])
    b['metadata']['source_sha256']='changed'
    with pytest.raises(ValueError): check_run(a,b,b['config'])


def test_seed_gate_detects_lossless_replay_failure():
    runs=rows()
    for r in runs:r['config']['seed']=43
    assert audit_seed(43,runs)['exact_p0_replay']['exact_replay']
    next(r for r in runs if r['config']['compression']=='qsgd-llz' and r['config']['llz-p']==0)['replay'][0]['model_sha256']='bad'
    with pytest.raises((ValueError,RuntimeError)):audit_seed(43,runs)
