"""Prepare verified CIFAR-10 records and deterministic federated splits."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory

import numpy as np


ARCHIVE_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
RECORD_BYTES = 1 + 3 * 32 * 32
COUNTS = {"train": 50_000, "test": 10_000}
BATCHES = {"train": [f"data_batch_{i}.bin" for i in range(1, 6)], "test": ["test_batch.bin"]}


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def unpack_batches(archive, directory):
    files = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for split, names in BATCHES.items():
            destination = directory / f"{split}.bin"
            with destination.open("wb") as output:
                for name in names:
                    member = bundle.getmember(f"cifar-10-batches-bin/{name}")
                    if not member.isfile() or member.size != 10_000 * RECORD_BYTES:
                        raise ValueError(f"Invalid CIFAR-10 archive member: {name}")
                    with bundle.extractfile(member) as stream:
                        shutil.copyfileobj(stream, output)
            files[split] = {"sha256": digest(destination), "records": COUNTS[split]}
    return files


def example(source, index, label, split, client):
    return {"filename": f"{source}:{index}", "identity": CLASSES[label], "label": label,
            "split": split, "client_id": client, "source": source, "index": index}


def split_examples(directory, clients, seed):
    rng = np.random.default_rng(seed)
    train = np.memmap(directory / "train.bin", dtype=np.uint8, mode="r",
                      shape=(COUNTS["train"], RECORD_BYTES))
    test = np.memmap(directory / "test.bin", dtype=np.uint8, mode="r",
                     shape=(COUNTS["test"], RECORD_BYTES))
    full, smoke = [], []
    try:
        for label in range(len(CLASSES)):
            ids = rng.permutation(np.flatnonzero(train[:, 0] == label))
            if len(ids) != 5_000:
                raise ValueError(f"Unexpected training class count for {CLASSES[label]}")
            for split, selected in (("train", ids[500:]), ("validation", ids[:500])):
                for offset, index in enumerate(selected):
                    row = example("train", int(index), label, split, (offset + label) % clients)
                    full.append(row)
                    if offset < 2 * clients:
                        smoke.append(row)
        labels = test[:, 0]
        if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 1_000)):
            raise ValueError("Unexpected official CIFAR-10 test class counts")
        full.extend(example("test", i, int(label), "test", None) for i, label in enumerate(labels))
    finally:
        del train, test
    return full, smoke


def prepare(archive, output, clients=4, seed=42):
    archive, output = Path(archive), Path(output)
    if not 1 <= clients <= 250 or not 0 <= seed < 2**32:
        raise ValueError("Use 1–250 clients and a seed in [0, 2**32)")
    if output.exists():
        raise FileExistsError(f"CIFAR-10 destination already exists: {output}")
    if digest(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("CIFAR-10 archive checksum differs from the publisher's binary archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".cifar10-") as temporary:
        staged = Path(temporary) / "data"
        staged.mkdir()
        files = unpack_batches(archive, staged)
        full, smoke = split_examples(staged, clients, seed)
        common = {"identities": CLASSES, "num_clients": clients, "split_seed": seed,
                  "archive_md5": ARCHIVE_MD5, "cifar10_files": files}
        for filename, variant, rows in (("manifest.json", "cifar10-iid", full),
                                        ("smoke-manifest.json", "cifar10-iid-smoke", smoke)):
            document = dict(common, image_variant=variant, examples=rows)
            (staged / filename).write_text(json.dumps(document, separators=(",", ":")) + "\n",
                                           encoding="utf-8")
        staged.rename(output)
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare(args.archive, args.output, args.clients, args.seed))


if __name__ == "__main__":
    main()
