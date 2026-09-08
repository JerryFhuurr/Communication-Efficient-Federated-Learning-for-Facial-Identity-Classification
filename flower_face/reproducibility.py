"""Stable model/source fingerprints for comparing runs on the same environment."""

import hashlib
import json
from pathlib import Path
import tomllib

import torch


def model_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        header = json.dumps([name, str(value.dtype), list(value.shape)]).encode()
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        digest.update(value.tobytes())
    return digest.hexdigest()


def source_hash(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    files = [root/'pyproject.toml']
    for package in ('flower_face', 'compression'):
        files.extend((root/package).rglob('*.py'))
    entries = {}
    for path in sorted(files):
        if path.is_file():
            # Flower rewrites TOML whitespace during packaging. Hash its values,
            # while keeping Python files byte-exact.
            content = (json.dumps(tomllib.loads(path.read_text()), sort_keys=True).encode()
                       if path.name == 'pyproject.toml' else path.read_bytes())
            entries[path.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def settings():
    return dict(protocol='partition-order-v1', aggregation_order='ascending dataset partition-id',
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                torch_threads=torch.get_num_threads(), cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                scope='Same source, data, software, and hardware; no cross-platform bitwise guarantee')
