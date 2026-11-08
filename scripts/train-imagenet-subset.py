"""Train with a materialized ImageNet subset in the pinned upstream sampler order."""

import argparse
from pathlib import Path
import runpy
import sys


UPSTREAM = Path("/upstream")


def restore_manifest_order(dataset, manifest):
    reader = dataset.reader
    root = Path(reader.root)
    remaining = {Path(path).relative_to(root).as_posix(): (path, label) for path, label in reader.samples}
    ordered = []
    with manifest.open(encoding="utf-8") as rows:
        for line_number, line in enumerate(rows, 1):
            name = line.rstrip("\n")
            try:
                ordered.append(remaining.pop(name))
            except KeyError:
                raise ValueError(f"Manifest line {line_number} names a missing or repeated image: {name!r}") from None

    if remaining:
        raise ValueError(f"{len(remaining):,} training images are not listed in the manifest, including {next(iter(remaining))!r}")
    if not ordered:
        raise ValueError("The subset manifest is empty")
    reader.samples = ordered
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    sys.path.insert(0, str(UPSTREAM))
    from didicm.utils import data_utils
    from timm.data import ImageNetInfo, create_dataset

    # Check the compact tree with the same reader before initializing training.
    dataset = create_dataset("", root=str(args.data_root), split="train", is_training=True)
    expected_labels = {name: index for index, name in enumerate(ImageNetInfo().label_names())}
    if dataset.reader.class_to_idx != expected_labels:
        parser.error("expected the canonical ImageNet-1k training label mapping")
    try:
        restore_manifest_order(dataset, args.manifest)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    del dataset

    def use_materialized_subset(dataset, data_ratio, seed=42):
        if data_ratio != 0.25 or seed != 42:
            raise ValueError("The compact ImageNet subset requires data ratio 0.25 and subset seed 42")
        return restore_manifest_order(dataset, args.manifest)

    data_utils.change_dataset_to_subset_for_data_ratio = use_materialized_subset
    sys.argv[1:] = ["--data-dir", str(args.data_root), *args.training_args]
    runpy.run_path(str(UPSTREAM / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
