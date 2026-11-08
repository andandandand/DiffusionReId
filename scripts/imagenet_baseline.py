"""Evaluate timm's published ResNet-50 A1 checkpoint with DiDiCM's validator."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import timm
from timm.data import ImageNetInfo
import torch
import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    # A full ImageNet result requires the original 1,000-way validation set.
    classes = sorted(path for path in args.validation.iterdir() if path.is_dir())
    if [path.name for path in classes] != ImageNetInfo().label_names() or any(
        sum(1 for image in directory.iterdir() if image.suffix.lower() in {".jpg", ".jpeg"})
        != 50
        for directory in classes
    ):
        parser.error("expected an ImageNet-1k ImageFolder with 50 images per synset")

    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "metrics.json"
    if result_path.exists():
        parser.error("metrics.json already exists in this output directory")
    model = "resnet50.a1_in1k"
    config = {
        "dataset": "ILSVRC2012",
        "split": "validation",
        "examples": 50000,
        "model": model,
        "weights": "timm/resnet50.a1_in1k",
        "paper_top1_pct": 80.42,
        "paper_top5_pct": 94.60,
        "checkpoint_reference_top1_pct": 80.38,
        "checkpoint_reference_top5_pct": 94.60,
        "image_size": 224,
        "crop_pct": 0.95,
        "batch_size": 64,
        "precision": "fp32",
        "upstream_commit": "7392557b5e40274a69e797cd910d42b50310f0ac",
        "torch": str(torch.__version__),
        "timm": timm.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
    }
    with wandb.init(
        name=args.output.name,
        job_type="checkpoint-eval",
        group="imagenet-paper-reproduction",
        tags=["imagenet-1k", "resnet50", "released-weights"],
        dir=str(args.output),
        config=config,
        save_code=False,
        settings=wandb.Settings(console="off", disable_git=True),
    ) as run:
        start = time.monotonic()
        subprocess.run(
            [
                sys.executable,
                str(args.upstream / "validate.py"),
                "--data-dir", str(args.validation),
                "--model", model,
                "--pretrained",
                "--img-size", str(config["image_size"]),
                "--crop-pct", str(config["crop_pct"]),
                "--use-train-size",
                "--device", args.device,
                "--batch-size", str(config["batch_size"]),
                "--workers", "4",
                "--log-freq", "50",
                "--results-file", str(result_path),
                "--results-format", "json",
            ],
            check=True,
        )
        results = json.loads(result_path.read_text())
        results.update(
            elapsed_seconds=round(time.monotonic() - start, 2),
            wandb_run_id=run.id,
        )
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        run.log(
            {
                "validation/top1_pct": results["top1"],
                "validation/top5_pct": results["top5"],
                "validation/elapsed_seconds": results["elapsed_seconds"],
            }
        )


if __name__ == "__main__":
    main()
