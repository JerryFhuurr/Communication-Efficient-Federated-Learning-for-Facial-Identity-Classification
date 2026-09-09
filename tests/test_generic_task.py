import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from flwr.app import Context, RecordDict
from flower_face.prepare_generic import prepare
from flower_face.task_api import resolve
from flower_face import client_app, server_app
from flower_face.run import default_config
from flower_face.experiment_suite import expand


def test_configuration_grid_is_explicit_and_keeps_task_settings():
    cells=expand({'grid':{'qsgd-levels':[31,127],'llz-window':[32,128]},
                  'task':{'num-classes':3},'study':{'seeds':[42]}})
    assert len(cells)==4
    assert {(c['study']['qsgd-levels'],c['study']['llz-window']) for c in cells} == {(31,32),(31,128),(127,32),(127,128)}
    assert all(c['task']['num-classes']==3 for c in cells)
    with pytest.raises(ValueError):
        expand({'grid':{'qsgd-levels':[31,31]}})


def test_suite_dispatch_and_resume_preserve_cell_protocol(tmp_path, monkeypatch):
    import sys
    from flower_face import experiment_suite as suite
    monkeypatch.setattr(suite,'__file__',str(tmp_path/'flower_face'/'experiment_suite.py'))
    monkeypatch.setattr(suite,'source_hash',lambda root:'frozen')
    config=tmp_path/'spec.toml'
    config.write_text('[study]\nseeds=[42]\nrounds=2\n')
    calls=[]
    def launch(command, cwd):
        calls.append(command)
        if '--output-dir' in command:
            destination=Path(command[command.index('--output-dir')+1])/'compression-study-test'
            destination.mkdir()
            (destination/'study.json').write_text('{}')
        return 0
    monkeypatch.setattr(suite.subprocess,'call',launch)
    monkeypatch.setattr(sys,'argv',['suite','--config',str(config)])
    suite.main()
    output=next((tmp_path/'outputs').iterdir())
    before=json.loads((output/'suite.json').read_text())
    monkeypatch.setattr(sys,'argv',['suite','--resume',str(output)])
    suite.main()
    after=json.loads((output/'suite.json').read_text())
    assert before==after
    assert '--resume' in calls[-1]


@pytest.mark.parametrize('method,p', [('none',0),('qsgd',0),('qsgd-llz',0),('qsgd-llz',1)])
def test_second_task_actual_flower_handlers(tmp_path, method, p):
    manifest = tmp_path/'manifest.json'
    prepare(manifest, synthetic=True)
    config = default_config(Path(__file__).resolve().parents[1])
    config.update({'task-module':'flower_face.synthetic_task', 'manifest':str(manifest),
                   'num-classes':3, 'compression':method, 'llz-p':p,
                   'num-server-rounds':2, 'output-dir':str(tmp_path/'runs')})
    class Grid:
        def get_node_ids(self):
            return [1,2,3,4]
        def send_and_receive(self, messages, timeout):
            for message in messages:
                node = message.metadata.dst_node_id
                context = Context(run_id=1, node_id=node,
                    node_config={'partition-id':node-1, 'num-partitions':4},
                    state=RecordDict(), run_config=config)
                yield client_app.app(message, context)
    server_app.main(Grid(), Context(run_id=1, node_id=0, node_config={},
                                   state=RecordDict(), run_config=config))
    run = next((tmp_path/'runs').iterdir())
    metrics = json.loads((run/'metrics.json').read_text())
    assert metrics['test'] is None
    assert metrics['communication']['total']['messages'] == 32
    assert metrics['communication']['total']['error_messages'] == 0
    assert all(torch.isfinite(x).all() for x in torch.load(run/'final_model.pt', weights_only=True)['state_dict'].values())


def test_prepare_class_folders(tmp_path):
    from PIL import Image
    for label in ('a','b'):
        folder=tmp_path/'images'/label
        folder.mkdir(parents=True)
        for i in range(16):
            Image.new('RGB',(8,8),(i, 100 if label=='a' else 200, 50)).save(folder/f'{i}.png')
    manifest=tmp_path/'custom.json'
    data=prepare(manifest, images=tmp_path/'images')
    assert len(data['examples']) == 32
    assert len({r['filename'] for r in data['examples']}) == 32
    config={'manifest':str(manifest),'num-classes':2,'num-clients':4,'image-size':32,
            'batch-size':4,'seed':42}
    task=resolve(config)
    x,y=next(iter(task.load_data(config,0)))
    assert x.shape[1:] == (3,32,32)
    assert task.Net(2)(x).shape == (len(y),2)
    row = next(r for r in data['examples'] if r['split']=='train' and r['client_id']==0)
    Image.new('RGB',(8,8),(255,255,255)).save(tmp_path/'images'/row['filename'])
    with pytest.raises(ValueError, match='Image changed'):
        list(task.load_data(config,0))


def test_task_preflight_rejects_integer_buffers(tmp_path, monkeypatch):
    from flower_face.task_api import validate
    from flower_face import synthetic_task
    path=tmp_path/'manifest.json'
    prepare(path, synthetic=True)
    class IntegerNet(torch.nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.register_buffer('counter', torch.tensor(0))
    monkeypatch.setattr(synthetic_task, 'Net', IntegerNet)
    config=default_config(Path(__file__).resolve().parents[1])
    config.update({'task-module':'flower_face.synthetic_task', 'manifest':str(path),
                   'num-classes':3, 'compression':'qsgd'})
    with pytest.raises(ValueError, match='Integer buffers'):
        validate(config)
