#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

mode="${1:?Usage: run-didicm-pilot.sh IMAGENET_ROOT|--smoke [RUN_ID]}"
run_id="${2:-didicm-pilot-$(date -u +%Y%m%dT%H%M%SZ)}"
case "$run_id" in
    ""|*[!a-zA-Z0-9._-]*|.|..) printf '%s\n' "Invalid run ID." >&2; exit 2 ;;
esac
upstream_commit=7392557b5e40274a69e797cd910d42b50310f0ac
test "$(git -C upstream/didicm rev-parse HEAD)" = "$upstream_commit"
test -z "$(git -C upstream/didicm status --porcelain)"
test -r .env
mkdir -p runs
mkdir "runs/$run_id"

if [[ "$mode" == "--smoke" ]]; then
    resources=(--cpus 2 --cpu-shares 128 --memory 6g)
    mounts=()
    args=(--device cpu --no-prefetcher --batch-size 8 --grad-accum-steps 2
        --validation-batch-size 8 --workers 1 --wandb-tags synthetic-smoke imagenet-shape)
else
    train="$(realpath "$mode/train")"
    validation="$(realpath "$mode/val")"
    for split in "$train" "$validation"; do
        if [[ $(find "$split" -mindepth 1 -maxdepth 1 -type d -name 'n????????' | wc -l) -ne 1000 ]]; then
            printf 'Expected 1,000 ImageNet synset directories under %s\n' "$split" >&2
            exit 2
        fi
    done
    val_count="$(find "$validation" -type f -iname '*.jpeg' | wc -l)"
    resources=(--gpus all)
    mounts=(
        --mount "type=bind,src=$train,dst=/data/train,readonly"
        --mount "type=bind,src=$validation,dst=/data/val,readonly"
    )
    manifest="$mode/train-subset-25.txt"
    if [[ -e "$manifest" || -L "$manifest" ]]; then
        if [[ ! -f "$manifest" || ! -r "$manifest" || "$val_count" -ne 50000 ]]; then
            printf 'Compact mode requires a readable train-subset-25.txt and 50,000 validation JPEGs.\n' >&2
            exit 2
        fi
        mounts+=(
            --mount "type=bind,src=$(realpath "$manifest"),dst=/train-subset-25.txt,readonly"
            --mount "type=bind,src=$PWD/scripts/train-imagenet-subset.py,dst=/train-imagenet-subset.py,readonly"
        )
    else
        train_count="$(find "$train" -type f -iname '*.jpeg' | wc -l)"
        if [[ "$train_count" -ne 1281167 || "$val_count" -ne 50000 ]]; then
            printf 'Expected 1,281,167 train and 50,000 validation JPEGs, found %s and %s.\n' \
                "$train_count" "$val_count" >&2
            exit 2
        fi
    fi
    args=(--device cuda --amp --batch-size 32 --grad-accum-steps 32
        --validation-batch-size 16 --workers 4 --recovery-interval 50
        --wandb-tags imagenet-1k single-epoch-pilot)
fi

exec docker run --rm --pull=never --name "diffusion-reid-$run_id" \
    "${resources[@]}" "${mounts[@]}" --user "$(id -u):$(id -g)" --shm-size 1g \
    --env HOME=/tmp --env OMP_NUM_THREADS=2 --env WANDB_DISABLE_GIT=true \
    --mount "type=bind,src=$PWD/.env,dst=/run/secrets/wandb.env,readonly" \
    --mount "type=bind,src=$PWD/upstream/didicm,dst=/upstream,readonly" \
    --mount "type=bind,src=$PWD/runs/$run_id,dst=/results" \
    --entrypoint bash diffusion-reid:paper-reproduction -c '
        set -euo pipefail
        set -a
        . /run/secrets/wandb.env
        set +a
        export WANDB_DIR=/results WANDB_JOB_TYPE=training-pilot
        dataset=/data
        if [[ "$1" == "--smoke" ]]; then
            dataset="$(mktemp -d)"
            python - "$dataset" <<"PY"
from pathlib import Path
import sys

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
rng = np.random.default_rng(42)
for split, count in (("train", 8), ("val", 2)):
    for label in range(8):
        folder = root / split / f"class-{label}"
        folder.mkdir(parents=True)
        for sample in range(count):
            pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{sample}.jpg")
PY
        fi
        shift
        run_id="$1"
        shift
        trainer=(/upstream/train.py --data-dir "$dataset")
        if [[ -f /train-subset-25.txt ]]; then
            trainer=(/train-imagenet-subset.py /train-subset-25.txt "$dataset")
        fi
        exec python -u "${trainer[@]}" \
            --input-size 3 56 56 --crop-pct 0.875 --model didirn50 --num-classes 1000 \
            --diffusion-enabled --diffusion-sampler cp --diffusion-steps 8 --diffusion-eps 1e-3 \
            --data-ratio 0.25 --epochs 1 --warmup-epochs 0 --opt lamb \
            --weight-decay 0.01 --sched cosine --lr 0.004 --seed 42 --log-interval 10 \
            --output /results --experiment "$run_id" --checkpoint-hist 1 \
            --log-wandb --wandb-project "$WANDB_PROJECT" "$@"
    ' _ "$mode" "$run_id" "${args[@]}"
