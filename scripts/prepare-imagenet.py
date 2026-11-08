"""Convert local ILSVRC/imagenet-1k publisher shards to an ImageFolder split."""

import argparse
from collections import Counter
import fcntl
from io import BytesIO
from pathlib import Path
import re
import shutil

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


NUM_CLASSES = 1000
# Publisher shard layout at revision 49e2ee26.
SPLITS = {"train": ("train", 294, 1_281_167), "validation": ("val", 14, 50_000)}
SCHEMA = pa.schema([
    ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("label", pa.int64()),
])
WNID = r"n[0-9]{8}"
PUBLISHER_NAMES = {
    "train": re.compile(rf"(?P<stem>(?P<prefix>{WNID})_[0-9]+)_(?P<wnid>{WNID})(?P<ext>(?i:\.jpeg))"),
    "validation": re.compile(rf"(?P<stem>ILSVRC2012_val_[0-9]{{8}})_(?P<wnid>{WNID})(?P<ext>(?i:\.jpeg))"),
}


def read_label_map(path):
    labels = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        label, sep, wnid = line.partition("\t")
        if label != str(index) or not sep or not re.fullmatch(WNID, wnid):
            raise ValueError(f"Invalid label map at line {index + 1}: expected {index}<TAB>wnid")
        labels.append(wnid)
    if len(labels) != NUM_CLASSES or labels != sorted(set(labels)):
        raise ValueError(f"Expected {NUM_CLASSES:,} unique synsets in ImageFolder class order")
    return labels


def check_shards(directory, split, count, expected_rows):
    shards = [directory / f"{split}-{index:05d}-of-{count:05d}.parquet" for index in range(count)]
    missing = [path.name for path in shards if not path.is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} of {count} {split} shards, starting with {missing[0]}")
    unexpected = set(directory.glob(f"{split}-*.parquet")) - set(shards)
    if unexpected:
        raise ValueError(f"Unexpected {split} shard: {min(unexpected).name}")
    rows = 0
    for path in shards:
        with pq.ParquetFile(path) as shard:
            if not shard.schema_arrow.equals(SCHEMA) or shard.metadata.num_rows == 0:
                raise ValueError(f"Unexpected schema or empty shard: {path.name}")
            rows += shard.metadata.num_rows
    if rows != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} {split} rows, found {rows:,}")
    return shards


def image_from_row(row, labels, split):
    image, label = row["image"], row["label"]
    if label is None or not 0 <= label < len(labels):
        raise ValueError(f"Invalid numeric label: {label}")
    if image is None or not image["path"] or not image["bytes"]:
        raise ValueError("Missing image path or bytes")
    match = PUBLISHER_NAMES[split].fullmatch(image["path"])
    if match is None:
        raise ValueError(f"Invalid publisher image path: {image['path']!r}")
    wnid = labels[label]
    if match["wnid"] != wnid or match.groupdict().get("prefix", wnid) != wnid:
        raise ValueError(f"Image path disagrees with label {label}: {image['path']}")
    # The publisher appends the WNID to each filename. Restore the archive name
    # so timm can enumerate files in the same order when sampling per class.
    return wnid, match["stem"] + match["ext"], image["bytes"]


def verify_image(data, name):
    # Input shards are expected to have passed publisher integrity checks.
    # Verify headers without decoding every image. Some original .JPEG files
    # can contain another format, so inspect the content rather than re-encode.
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format == "JPEG" and data.rfind(b"\xff\xd9") == -1:
                raise ValueError("missing JPEG end marker")
            image.verify()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Invalid image {name}: {exc}") from exc


def prepare(shard_dir, output_root, label_map, split, min_free_gib=0):
    if min_free_gib < 0:
        raise ValueError("--min-free-gib cannot be negative")
    min_free_bytes = min_free_gib * 1024**3
    folder, shard_count, expected_rows = SPLITS[split]
    labels = read_label_map(label_map)
    shards = check_shards(shard_dir, split, shard_count, expected_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / folder
    staging = output_root / f".{folder}.partial"
    temporary_image = output_root / f".{folder}.image.tmp"

    with (output_root / f".{folder}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = final.exists()
        target = final if complete else staging
        if final.is_symlink() or target.is_symlink():
            raise ValueError(f"Refusing a symlink for the {folder} split")
        target.mkdir(exist_ok=True)
        temporary_image.unlink(missing_ok=True)
        for wnid in labels:
            directory = target / wnid
            if directory.is_symlink():
                raise ValueError(f"Refusing a symlink for class {wnid}")
            if not complete:
                directory.mkdir(exist_ok=True)

        counts = Counter()
        seen = set()
        for path in shards:
            if min_free_bytes and not complete:
                free = shutil.disk_usage(output_root).free
                if free < min_free_bytes + path.stat().st_size:
                    raise ValueError(f"Insufficient free disk space for {path.name} with a {min_free_gib} GiB reserve")
            with pq.ParquetFile(path) as shard:
                for batch in shard.iter_batches(batch_size=64):
                    for row in batch.to_pylist():
                        wnid, name, data = image_from_row(row, labels, split)
                        if name in seen:
                            raise ValueError(f"Duplicate image path: {name}")
                        seen.add(name)
                        counts[wnid] += 1
                        verify_image(data, name)
                        destination = target / wnid / name
                        if destination.exists() or destination.is_symlink():
                            if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != data:
                                raise ValueError(f"Existing image differs: {destination}")
                        elif complete:
                            raise ValueError(f"Published split is missing {destination}")
                        else:
                            temporary_image.write_bytes(data)
                            temporary_image.replace(destination)
            print(f"Checked {path.name} ({len(seen):,}/{expected_rows:,} images)", flush=True)

        if len(seen) != expected_rows or set(counts) != set(labels):
            raise ValueError("Image count or class coverage is incomplete")
        if split == "validation" and set(counts.values()) != {expected_rows // NUM_CLASSES}:
            raise ValueError("Validation classes must have equal image counts")
        if {entry.name for entry in target.iterdir()} != set(labels) or any(
            sum(1 for _ in (target / wnid).iterdir()) != counts[wnid] for wnid in labels
        ):
            raise ValueError(f"Unexpected directories or files in {target}")
        if not complete:
            staging.rename(final)
        print(f"Ready: {final} ({len(seen):,} images, {len(labels):,} classes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path, help="Directory containing verified publisher Parquet files")
    parser.add_argument("output_root", type=Path, help="ImageFolder root under which to create train/ or val/")
    parser.add_argument("--label-map", type=Path, required=True, help="Numeric label to WordNet synset TSV")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--min-free-gib", type=int, default=0, help="Free disk reserve to check before each shard")
    args = parser.parse_args()
    try:
        prepare(args.shard_dir, args.output_root, args.label_map, args.split, args.min_free_gib)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
