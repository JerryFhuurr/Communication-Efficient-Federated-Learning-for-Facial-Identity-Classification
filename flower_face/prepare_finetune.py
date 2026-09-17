"""Cache the frozen ResNet prefix, preserving an existing feature manifest's split."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from flower_face.finetune_task import file_hash, verify
from flower_face.pretrained_task import read_manifest
from flower_face.task import seed_everything


def prepare(source_manifest, output, head_checkpoint=None, batch_size=16):
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    source_manifest, output = Path(source_manifest).resolve(), Path(output).resolve()
    raw = json.loads(source_manifest.read_text())
    manifest = read_manifest(source_manifest, len(raw['identities']), raw['num_clients'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    meta = manifest['feature_extractor']
    weights_path = verify(source_manifest.parent / meta['weights_file'], meta['weights_sha256'])
    saved = None
    if head_checkpoint is not None:
        saved = torch.load(head_checkpoint, map_location='cpu', weights_only=True)
        if (saved['manifest_sha256'] != file_hash(source_manifest)
                or saved['identities'] != manifest['identities']
                or saved['config']['task-module'] != 'flower_face.pretrained_task'):
            raise ValueError('Warm-start head must match this frozen-feature manifest')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, output / meta['weights_file'])
    if saved is not None:
        shutil.copy2(head_checkpoint, output / 'initial_head.pt')
        manifest['initial_head'] = dict(file='initial_head.pt', sha256=file_hash(output / 'initial_head.pt'),
            source_checkpoint=str(Path(head_checkpoint).resolve()), source_seed=saved['config']['seed'],
            source_manifest_sha256=saved['manifest_sha256'], scope='Shared validation-selected warm start')
    seed_everything(42)
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.requires_grad_(False).eval()
    prefix = nn.Sequential(*list(model.children())[:7])
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    rows = manifest['examples']
    array = np.lib.format.open_memmap(output / 'features.npy', mode='w+', dtype=np.float32,
                                    shape=(len(rows), 2, 256, 14, 14))
    old = np.load(source_manifest.parent / meta['file'], mmap_mode='r', allow_pickle=False)
    started = time.perf_counter()
    max_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            tensors = []
            for row in rows[start:start + batch_size]:
                path = Path(manifest['image_root']) / row['filename']
                verify(path, row['sha256'])
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                    tensors.extend((transform(rgb), transform(ImageOps.mirror(rgb))))
            intermediate = prefix(torch.stack(tensors))
            if not torch.isfinite(intermediate).all():
                raise ValueError('Nonfinite intermediate features')
            n = len(tensors) // 2
            array[start:start+n] = intermediate.numpy().reshape(n, 2, 256, 14, 14)
            # One implementation check within extraction, not an extra training run.
            if start == 0:
                reconstructed = model.avgpool(model.layer4(intermediate)).flatten(1).numpy().reshape(n, 2, 512)
                reference = old[:n]
                max_error = float(np.max(np.abs(reconstructed - reference)))
                if not np.allclose(reconstructed, reference, atol=2e-5, rtol=2e-5):
                    raise ValueError('Intermediate prefix does not reconstruct the original features')
            if start % (batch_size * 20) == 0:
                print(f'Cached {start+n}/{len(rows)} images', flush=True)
    array.flush()
    del array
    manifest['image_variant'] = 'frozen-resnet18-layer3-features'
    manifest['feature_extractor'] = dict(meta, stage='layer3', shape=[256,14,14],
        file='features.npy', sha256=file_hash(output / 'features.npy'),
        source_feature_manifest_sha256=file_hash(source_manifest),
        elapsed_seconds=time.perf_counter()-started, reconstruction_max_absolute_error=max_error,
        scope='Frozen prefix through layer3; layer4 and classifier are federated. '
              'Cache preparation, warm-start training, and prefix distribution excluded from round traffic.')
    manifest['feature_extractor'].pop('dimension', None)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Saved {output / "manifest.json"}; reconstruction error={max_error:g}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head-checkpoint', type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    prepare(args.source_manifest, args.output, args.head_checkpoint, args.batch_size)
