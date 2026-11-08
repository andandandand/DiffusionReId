"""Convert local ILSVRC/imagenet-1k publisher shards to an ImageFolder split."""

import argparse
from bisect import bisect_left
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
PILOT_MANIFEST = "train-subset-25.txt"


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


def image_location(path, label, labels, split):
    if label is None or not 0 <= label < len(labels):
        raise ValueError(f"Invalid numeric label: {label}")
    if not path:
        raise ValueError("Missing image path")
    match = PUBLISHER_NAMES[split].fullmatch(path)
    if match is None:
        raise ValueError(f"Invalid publisher image path: {path!r}")
    wnid = labels[label]
    if match["wnid"] != wnid or match.groupdict().get("prefix", wnid) != wnid:
        raise ValueError(f"Image path disagrees with label {label}: {path}")
    # The publisher appends the WNID to each filename. Restore the archive name
    # so timm can enumerate files in the same order when sampling per class.
    return wnid, match["stem"] + match["ext"]


def plan_pilot_subset(shards, labels):
    import torch
    from timm.utils.misc import natural_key

    classes = [[] for _ in labels]
    seen = set()
    for shard_index, path in enumerate(shards):
        row_index = 0
        with pq.ParquetFile(path) as shard:
            for batch in shard.iter_batches(batch_size=4096, columns=["image.path", "label"]):
                for row in batch.to_pylist():
                    image, label = row["image"], row["label"]
                    _, name = image_location(image["path"] if image else None, label, labels, "train")
                    if name in seen:
                        raise ValueError(f"Duplicate image path: {name}")
                    seen.add(name)
                    classes[label].append((name, shard_index, row_index))
                    row_index += 1

    selected_rows = [[] for _ in shards]
    manifest = []
    for wnid, samples in zip(labels, classes):
        samples.sort(key=lambda sample: natural_key(sample[0]))
        keep = len(samples) // 4
        if not keep:
            raise ValueError(f"Cannot keep 25% of class {wnid} with only {len(samples)} images")
        # DiDiCM resets its generator to seed 42 independently for every class.
        order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(42))[:keep]
        for index in order.tolist():
            name, shard_index, row_index = samples[index]
            selected_rows[shard_index].append(row_index)
            manifest.append(f"{wnid}/{name}")
    for indices in selected_rows:
        indices.sort()
    return selected_rows, manifest


def selected_batches(shard, indices):
    if not indices:
        return
    offset = 0
    first = 0
    for batch in shard.iter_batches(batch_size=64):
        end = offset + batch.num_rows
        last = bisect_left(indices, end, lo=first)
        if last > first:
            yield batch.take([index - offset for index in indices[first:last]])
        if last == len(indices):
            return
        first = last
        offset = end
    raise ValueError("Shard ended before the selected rows were available")


def publish_manifest(output_root, paths, complete):
    manifest = output_root / PILOT_MANIFEST
    contents = "\n".join(paths) + "\n"
    if manifest.is_symlink():
        raise ValueError(f"Refusing a symlink for {manifest}")
    if manifest.exists():
        if not manifest.is_file() or manifest.read_bytes() != contents.encode("utf-8"):
            raise ValueError(f"Existing pilot manifest differs: {manifest}")
    elif complete:
        raise ValueError(f"Published pilot split is missing {manifest}")
    else:
        temporary = output_root / f".{PILOT_MANIFEST}.tmp"
        temporary.unlink(missing_ok=True)
        temporary.write_text(contents, encoding="utf-8", newline="\n")
        temporary.replace(manifest)


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


def prepare(shard_dir, output_root, label_map, split, min_free_gib=0, pilot_subset=False):
    if min_free_gib < 0:
        raise ValueError("--min-free-gib cannot be negative")
    if pilot_subset and split != "train":
        raise ValueError("--pilot-subset requires --split train")
    min_free_bytes = min_free_gib * 1024**3
    folder, shard_count, expected_rows = SPLITS[split]
    labels = read_label_map(label_map)
    shards = check_shards(shard_dir, split, shard_count, expected_rows)
    pilot_paths = None
    selected_rows = None
    if pilot_subset:
        print("Planning the upstream 25% training subset from publisher metadata", flush=True)
        selected_rows, pilot_paths = plan_pilot_subset(shards, labels)
        expected_rows = len(pilot_paths)
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
        for shard_index, path in enumerate(shards):
            if min_free_bytes and not complete:
                free = shutil.disk_usage(output_root).free
                if free < min_free_bytes + path.stat().st_size:
                    raise ValueError(f"Insufficient free disk space for {path.name} with a {min_free_gib} GiB reserve")
            with pq.ParquetFile(path) as shard:
                batches = (shard.iter_batches(batch_size=64) if selected_rows is None
                           else selected_batches(shard, selected_rows[shard_index]))
                for batch in batches:
                    for row in batch.to_pylist():
                        image = row["image"]
                        wnid, name = image_location(image["path"] if image else None, row["label"], labels, split)
                        data = image["bytes"]
                        if not data:
                            raise ValueError(f"Missing image bytes for {name}")
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
        if pilot_paths is not None:
            if seen != {path.split("/", 1)[1] for path in pilot_paths}:
                raise ValueError("Extracted images disagree with the planned pilot subset")
            publish_manifest(output_root, pilot_paths, complete)
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
    parser.add_argument("--pilot-subset", action="store_true", help="Stage the exact upstream 25%% training subset")
    args = parser.parse_args()
    try:
        prepare(args.shard_dir, args.output_root, args.label_map, args.split, args.min_free_gib, args.pilot_subset)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
