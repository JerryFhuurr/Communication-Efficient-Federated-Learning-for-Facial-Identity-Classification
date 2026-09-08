"""Build a reproducible, disjoint, balanced subset from local CelebA images."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from PIL import Image


def select_examples(identity_file, num_identities=10, images_per_identity=30, seed=42):
    """Select identities and images independently of directory iteration order."""
    if num_identities < 2 or images_per_identity < 12:
        raise ValueError("Use at least 2 identities and 12 images per identity.")
    groups = defaultdict(list)
    for line in Path(identity_file).read_text().splitlines():
        name, identity = line.split()
        if Path(name).name != name or not name.endswith(".jpg"):
            raise ValueError(f"Unexpected image filename: {name}")
        groups[int(identity)].append(name)
    eligible = sorted(k for k, v in groups.items() if len(v) >= images_per_identity)
    if len(eligible) < num_identities:
        raise ValueError(f"Only {len(eligible)} identities have {images_per_identity} images.")
    rng = random.Random(seed)
    identities = sorted(rng.sample(eligible, num_identities))
    examples = []
    # Four validation images (one per client), roughly 20% held out for test.
    num_test = max(4, round(images_per_identity * 0.2))
    num_train = images_per_identity - 4 - num_test
    if num_train < 4:
        raise ValueError("Too few training images for four clients.")
    for label, identity in enumerate(identities):
        selected = rng.sample(sorted(groups[identity]), images_per_identity)
        for index, name in enumerate(selected):
            if index < num_train:
                split, client = "train", index % 4
            elif index < num_train + 4:
                split, client = "validation", (index - num_train) % 4
            else:
                split, client = "test", None
            examples.append(dict(filename=name, identity=identity, label=label,
                                 split=split, client_id=client))
    return identities, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, help="Directory containing CelebA JPGs")
    parser.add_argument("--identity-file", type=Path, default=Path("identity_CelebA.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/subset/manifest.json"))
    parser.add_argument("--num-identities", type=int, default=10)
    parser.add_argument("--images-per-identity", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-unaligned", action="store_true",
                        help="Explicitly accept originals; resizing is not face alignment")
    parser.add_argument("--archive-list", type=Path,
                        help="Only write a 7-Zip include list for selected images")
    args = parser.parse_args()
    identities, examples = select_examples(args.identity_file, args.num_identities,
                                           args.images_per_identity, args.seed)
    if args.archive_list:
        args.archive_list.parent.mkdir(parents=True, exist_ok=True)
        args.archive_list.write_text("\n".join("img_celeba/" + r["filename"] for r in examples))
        print(f"Wrote {len(examples)} selected archive paths to {args.archive_list}")
        return
    if args.images is None:
        parser.error("--images is required unless using --archive-list")
    root = args.images.resolve()
    for row in examples:
        path = root / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected image: {path}")
        with Image.open(path) as im:
            if not args.allow_unaligned and im.size != (178, 218):
                raise ValueError(f"{path} has size {im.size}; expected aligned CelebA 178x218. "
                                 "Use the aligned download or explicitly pass --allow-unaligned.")
            im.verify()
    manifest = dict(version=1, image_root=str(root), identities=identities,
                    num_clients=4, seed=args.seed,
                    image_variant="original-unaligned" if args.allow_unaligned else "aligned-cropped",
                    split_method="seeded stratified closed-set split; official partition unused",
                    identity_file_sha256=hashlib.sha256(args.identity_file.read_bytes()).hexdigest(),
                    examples=examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    counts = {s: sum(r["split"] == s for r in examples) for s in ("train", "validation", "test")}
    print(f"Prepared {len(identities)} identities: {counts}")
    print(f"Image variant: {manifest['image_variant']}; manifest: {args.output.resolve()}")


if __name__ == "__main__":
    main()
