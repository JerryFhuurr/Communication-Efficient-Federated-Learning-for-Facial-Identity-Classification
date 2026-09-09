"""Create disjoint IID manifests from class folders or deterministic vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def prepare(output, *, images=None, clients=4, seed=42, synthetic=False):
    if clients < 1:
        raise ValueError('clients must be positive')
    rng = np.random.default_rng(seed)
    rows, hashes = [], set()
    if synthetic:
        classes = ['class-0', 'class-1', 'class-2']
        centers = rng.normal(size=(3, 8)) * 2
        groups = [[(f'{label}-{i}', None) for i in range(clients*12)] for label in range(3)]
    else:
        images = Path(images).resolve()
        folders = sorted(p for p in images.iterdir() if p.is_dir())
        classes = [p.name for p in folders]
        groups = [[(p.relative_to(images).as_posix(), p) for p in sorted(folder.rglob('*'))
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}] for folder in folders]
    if len(classes) < 2:
        raise ValueError('At least two classes required')
    for label, files in enumerate(groups):
        if len(files) < clients*3:
            raise ValueError(f'{classes[label]} needs at least {clients*3} images')
        order = rng.permutation(len(files))
        validation = max(clients, len(files)//5)
        test_count = max(1, len(files)//5)
        counts = [('validation', validation), ('test', test_count),
                  ('train', len(files)-validation-test_count)]
        offset = 0
        for split, count in counts:
            for i in range(count):
                name, path = files[order[offset+i]]
                row = dict(filename=name, label=label, identity=classes[label], split=split,
                           client_id=i % clients if split != 'test' else None)
                if synthetic:
                    row['features'] = (centers[label]+rng.normal(size=8)).tolist()
                else:
                    with Image.open(path) as image:
                        pixels = image.convert('RGB')
                        digest = hashlib.sha256(str(pixels.size).encode()+pixels.tobytes()).hexdigest()
                    if digest in hashes:
                        raise ValueError(f'Duplicate image content: {path}')
                    hashes.add(digest)
                    row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append(row)
            offset += count
    result = dict(identities=classes, num_clients=clients, examples=rows, seed=seed,
                  image_root=str(images) if not synthetic else '',
                  image_variant='synthetic-vectors' if synthetic else 'generic-class-folders')
    output = Path(output)
    if output.exists():
        raise ValueError('Manifest already exists; use a new output filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--images', type=Path)
    source.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clients', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.output, images=args.images, synthetic=args.synthetic,
                     clients=args.clients, seed=args.seed)
    print(f"Saved {len(result['examples'])} examples, {len(result['identities'])} classes to {args.output}")


if __name__ == '__main__':
    main()
