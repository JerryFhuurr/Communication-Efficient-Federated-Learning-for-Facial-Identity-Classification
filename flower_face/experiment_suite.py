"""Run a configuration grid as isolated, resumable matched compression studies."""
import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import tomli_w
from flower_face.reproducibility import source_hash
from flower_face.study import write_json


def expand(spec):
    if set(spec)-{'study','task','grid'}:
        raise ValueError('Unknown specification table')
    grid = spec.get('grid', {})
    if set(grid)-{'qsgd-levels','llz-window'}:
        raise ValueError('Grid supports qsgd-levels and llz-window')
    levels=grid.get('qsgd-levels',[spec.get('study',{}).get('qsgd-levels',127)])
    windows=grid.get('llz-window',[spec.get('study',{}).get('llz-window',128)])
    for values in (levels,windows):
        if not values or len(set(values))!=len(values) or any(type(x) is not int or not 1<=x<=65535 for x in values):
            raise ValueError('Grid values must be distinct integers in [1,65535]')
    return [dict(study=dict(spec.get('study',{}), **{'qsgd-levels':s,'llz-window':w}),
                 task=spec.get('task',{})) for s,w in product(levels,windows)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    if bool(args.config)==bool(args.resume):
        parser.error('Choose --config or --resume')
    root=Path(__file__).resolve().parents[1]
    if args.config:
        cells=expand(tomllib.loads(args.config.read_text()))
        if args.plan_only:
            print(json.dumps(cells,indent=2)); return
        output=root/'outputs'/('suite-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        output.mkdir(parents=True)
        record=dict(status='running', source_sha256=source_hash(root), cells=cells, completed=[])
        write_json(output/'suite.json',record)
    else:
        output=args.resume.resolve()
        record=json.loads((output/'suite.json').read_text())
        if source_hash(root)!=record['source_sha256']:
            raise ValueError('Suite source changed; restore its source version before resuming')
    print(f'Suite directory: {output}',flush=True)
    try:
        for i,cell in enumerate(record['cells']):
            directory=output/f'cell-{i}'
            directory.mkdir(exist_ok=True)
            path=directory/'config.toml'
            path.write_text(tomli_w.dumps(cell))
            existing=list(directory.glob('compression-study-*/study.json'))
            if len(existing)>1:
                raise ValueError('Ambiguous cell studies')
            command=[sys.executable,'-m','flower_face.compression_study']
            # Resume also revalidates already completed cell artifacts.
            command += ['--resume',str(existing[0].parent)] if existing else ['--config',str(path),'--output-dir',str(directory)]
            print(f'Cell {i+1}/{len(record["cells"])}: {cell["study"]}',flush=True)
            if subprocess.call(command,cwd=root)!=0:
                raise RuntimeError(f'Cell {i} failed; suite can be resumed')
            if i not in record['completed']:
                record['completed'].append(i)
            write_json(output/'suite.json',record)
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=str(error)); raise
    finally:
        write_json(output/'suite.json',record)


if __name__=='__main__':
    main()
