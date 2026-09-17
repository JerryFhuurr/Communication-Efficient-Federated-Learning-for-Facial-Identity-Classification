"""Classify one cropped face with a frozen ResNet-18 and a saved FL head."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from flower_face.pretrained_task import Net
from flower_face.pretrained_mlp_task import Net as MLPHead


def load_predictor(checkpoint, manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('Checkpoint and feature manifest do not match')
    if saved['identities'] != manifest['identities']:
        raise ValueError('Checkpoint identity mapping differs from manifest')
    heads = {'flower_face.pretrained_task': Net, 'flower_face.pretrained_mlp_task': MLPHead}
    if saved['config']['task-module'] not in heads:
        raise ValueError('Expected a frozen ResNet-18 head checkpoint')
    meta = manifest['feature_extractor']
    if meta['weights'] != 'ResNet18_Weights.IMAGENET1K_V1' or meta['dimension'] != 512:
        raise ValueError('Unsupported feature extractor')
    weights_path = manifest_path.parent / meta['weights_file']
    if hashlib.sha256(weights_path.read_bytes()).hexdigest() != meta['weights_sha256']:
        raise ValueError('Backbone weights differ from feature preparation')
    # Preserve the caller's RNG state: constructor random draws are discarded.
    with torch.random.fork_rng(devices=[]):
        backbone = resnet18(weights=None)
        backbone.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        backbone.fc = nn.Identity()
        head = heads[saved['config']['task-module']](len(manifest['identities']))
        head.load_state_dict(saved['state_dict'])
    model = nn.Sequential(backbone, head).requires_grad_(False).eval()
    return model, ResNet18_Weights.IMAGENET1K_V1.transforms(), manifest['identities']


def predict(checkpoint, manifest, image_path, top_k=5):
    if type(top_k) is not int or top_k < 1:
        raise ValueError('top-k must be positive')
    model, preprocess, identities = load_predictor(checkpoint, manifest)
    with Image.open(image_path) as image:
        inputs = preprocess(image.convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        values, indices = model(inputs).softmax(dim=1)[0].topk(min(top_k, len(identities)))
    return [dict(identity=identities[i], label=i, softmax_score=float(value))
            for value, i in zip(values.tolist(), indices.tolist())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(dict(predictions=predict(args.checkpoint, args.manifest, args.image, args.top_k),
                         scope='Closed-set identity classification; softmax scores are not calibrated confidence. '
                               'This command does not detect/crop faces or reject unknown identities.'), indent=2))


if __name__ == '__main__':
    main()
