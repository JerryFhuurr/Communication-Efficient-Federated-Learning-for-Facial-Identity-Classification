from copy import deepcopy
import json

import pytest

from flower_face import adaptive_study as study
from flower_face import run as launcher
from test_compression_study import run


def test_budget_rule_uses_saved_errors_and_keeps_two_distinct_positive_limits():
    rows=[{'relative_errors':[0.,v,4*v]} for v in (.011,.021,.031,.041)]
    budgets=study.choose_budgets(rows)
    assert budgets['tight']==.018  # p1 Q25 = .0185, rounded down.
    assert budgets['loose']==.14   # p2 Q75 = .134, rounded up.
    assert study.choose_budgets([{'relative_errors':[0.,0.,.01]}])['tight']>0
    with pytest.raises(ValueError):
        study.choose_budgets([{'relative_errors':[0.,0.,0.]}])
    with pytest.raises(ValueError):
        study.choose_budgets([{'relative_errors':[0.,float('nan'),.1]}])


def fixture_record(tmp_path):
    config={'seed':42,'compression':'qsgd','output-dir':'unused','qsgd-levels':127,
            'llz-p':0,'llz-window':128,'llz-budget':0.,'num-server-rounds':15,'num-clients':4}
    record={'config':config,'budgets':{'tight':.01,'loose':.1},'runs':[]}
    for label in study.METHODS:
        folder=tmp_path/label;folder.mkdir()
        result=run(42,'qsgd',total=100000,upload=40000)
        result.update(label=label,output=str(folder),config=study.run_config(record,label,str(folder)))
        result['metadata'].update(python='same',fixed_state_policy='same')
        if label.startswith('adaptive-'):
            budget=result['config']['llz-budget']
            p=1 if label=='adaptive-tight' else 2
            rows=[];messages=[]
            for r in range(1,16):
                for c in range(4):
                    rows.append(dict(round=r,client=c,adaptive_selected_p=p,
                        qsgd_input_squared_norm=1.,llz_reconstruction_squared_error=[0.,.005,.05][p],
                        adaptive_candidate_squared_error=[0.,.005,.05],
                        adaptive_candidate_message_bytes=[1000.,800.,600.],
                        adaptive_candidate_payload_bytes=[900.,700.,500.],
                        adaptive_budget=budget,observed_message_bytes=[1000,800,600][p],
                        adaptive_selected_message_bytes=[1000.,800.,600.][p],
                        adaptive_selection_seconds=.1,decode_seconds=.01))
                    messages.append(dict(round=r,client_partition_id=c,phase='train',direction='uplink',
                        serialized_object_bytes=[1000,800,600][p],codec_payload_bytes=[900,700,500][p]))
            for name,values in [('adaptive_selection.jsonl',rows),('communication_messages.jsonl',messages)]:
                (folder/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
        record['runs'].append(result)
    return record


def test_audit_accepts_exact_p0_and_checks_all_adaptive_client_round_choices(tmp_path):
    record=fixture_record(tmp_path)
    summary=study.audit(record)
    assert summary['exact_p0_replay']['exact_replay']
    assert summary['methods']['adaptive-tight']['selected_p_counts']['1']==60
    assert summary['methods']['adaptive-loose']['selected_p_counts']['2']==60


@pytest.mark.parametrize('corruption',['budget','missing','byte-count','infeasible','replay'])
def test_audit_rejects_policy_and_replay_failures(tmp_path,corruption):
    record=fixture_record(tmp_path)
    if corruption=='budget':
        record['runs'][-1]['config']['llz-budget']=999.
    elif corruption=='replay':
        record['runs'][1]['replay'][0]['model_sha256']='wrong'
    else:
        path=tmp_path/'adaptive-tight'/'adaptive_selection.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        if corruption=='missing': rows.pop()
        elif corruption=='byte-count': rows[0]['observed_message_bytes']=999
        else: rows[0]['llz_reconstruction_squared_error']=1.
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises((ValueError,RuntimeError)):
        study.audit(record)


def test_streamed_launch_preserves_full_log_and_delivers_completion_lines(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'validate_task',lambda cfg:None)
    class Process:
        stdout=iter(['first line\n','Round 1 serialized objects: 99 bytes\n'])
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def wait(self):return 0
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**kw:Process())
    lines=[];log=tmp_path/'terminal.log'
    assert launcher.launch({'seed':42,'num-clients':4},root=tmp_path,log_path=log,progress=lines.append)==0
    assert log.read_text()=='first line\nRound 1 serialized objects: 99 bytes\n'
    assert lines==['first line','Round 1 serialized objects: 99 bytes']
