"""Cache fixed pretrained features, preserving image splits and identity labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import shutil
from urllib.request import urlretrieve
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from flower_face.task import read_manifest, seed_everything


def prepare(manifest_path, output, batch_size=32, weights_path=None):
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output exists; use a new directory')
    raw = json.loads(Path(manifest_path).read_text())
    manifest = read_manifest(manifest_path, len(raw['identities']), raw['num_clients'])
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    weights = ResNet18_Weights.IMAGENET1K_V1
    path = output/'resnet18-f37072fd.pth'
    started = time.perf_counter()
    if weights_path is None:
        print('Downloading official ResNet-18 weights (about 45 MB)', flush=True)
        urlretrieve(weights.url, path)
    else:
        shutil.copy2(weights_path, path)
    weights_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if not weights_hash.startswith('f37072fd'):
        raise ValueError('Pretrained weights checksum mismatch')
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.fc = nn.Identity()
    model.requires_grad_(False).eval()
    preprocess = weights.transforms()
    rows = manifest['examples']
    array = np.empty((len(rows),2,512),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            tensors=[]
            for row in rows[start:start+batch_size]:
                image_path = Path(manifest['image_root'])/row['filename']
                if hashlib.sha256(image_path.read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError(f'Source image changed: {image_path}')
                with Image.open(image_path) as image:
                    rgb=image.convert('RGB')
                    tensors.extend((preprocess(rgb),preprocess(ImageOps.mirror(rgb))))
            vectors=model(torch.stack(tensors)).numpy().reshape(-1,2,512)
            array[start:start+len(vectors)]=vectors
            if start % (batch_size*10) == 0:
                print(f'Extracted {start+len(vectors)}/{len(rows)} images',flush=True)
    np.save(output/'features.npy',array,allow_pickle=False)
    manifest['image_variant']='frozen-resnet18-imagenet1k-v1-features'
    manifest['feature_extractor']=dict(
        weights='ResNet18_Weights.IMAGENET1K_V1', weights_url=weights.url,
        weights_sha256=weights_hash, weights_file=path.name, weights_bytes=path.stat().st_size,
        backbone_parameters=sum(p.numel() for p in model.parameters()), dimension=512,
        file='features.npy', sha256=hashlib.sha256((output/'features.npy').read_bytes()).hexdigest(),
        preprocessing=str(preprocess), views=['original','horizontal-flip'],
        source_manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started,
        scope='Frozen external ImageNet weights; no dataset fitting. Head-only federated training. '
              'Feature preparation and weight distribution are outside Flower traffic accounting.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"Saved {output/'manifest.json'}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--weights',type=Path,help='Local official weights file; checksum is verified')
    args=parser.parse_args()
    prepare(args.manifest,args.output,args.batch_size,args.weights)
