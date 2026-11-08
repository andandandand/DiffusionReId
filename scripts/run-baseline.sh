#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

validation="$(realpath "${1:?Usage: run-baseline.sh VALIDATION_IMAGEFOLDER [cpu|cuda] [RUN_ID]}")"
device="${2:-cuda}"
run_id="${3:-resnet50-a1-$(date -u +%Y%m%dT%H%M%SZ)}"
case "$run_id" in
    ""|*[!a-zA-Z0-9._-]*|.|..) printf '%s\n' "Invalid run ID." >&2; exit 2 ;;
esac
case "$device" in
    cuda) resources=(--gpus all) ;;
    cpu) resources=(--cpus 4 --cpu-shares 256 --memory 8g) ;;
    *) printf '%s\n' "Device must be cpu or cuda." >&2; exit 2 ;;
esac
upstream_commit=7392557b5e40274a69e797cd910d42b50310f0ac
test "$(git -C upstream/didicm rev-parse HEAD)" = "$upstream_commit"
test -z "$(git -C upstream/didicm status --porcelain)"
test -r .env
mkdir -p runs/hf-cache
mkdir "runs/$run_id"
exec docker run --rm --pull=never --name "diffusion-reid-$run_id" \
    "${resources[@]}" --user "$(id -u):$(id -g)" --shm-size 1g \
    --env HOME=/tmp --env HF_HOME=/cache --env OMP_NUM_THREADS=4 \
    --mount "type=bind,src=$PWD/.env,dst=/run/secrets/wandb.env,readonly" \
    --mount "type=bind,src=$PWD/scripts,dst=/scripts,readonly" \
    --mount "type=bind,src=$PWD/upstream/didicm,dst=/upstream,readonly" \
    --mount "type=bind,src=$validation,dst=/data/val,readonly" \
    --mount "type=bind,src=$PWD/runs/hf-cache,dst=/cache" \
    --mount "type=bind,src=$PWD/runs/$run_id,dst=/results/$run_id" \
    --entrypoint bash diffusion-reid:paper-reproduction -c '
        set -euo pipefail
        set -a
        . /run/secrets/wandb.env
        set +a
        exec python -u /scripts/imagenet_baseline.py \
            --upstream /upstream --validation /data/val --device "$1" --output "$2"
    ' _ "$device" "/results/$run_id"
