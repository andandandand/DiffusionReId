# Experiment log

| Date (UTC) | Run | Status | W&B ID |
| --- | --- | --- | --- |
| 2026-11-08 | ImageNet-1k ResNet-50 A1, 224 px, 0.95 crop, FP32, Spark CPU | Running on all 50,000 validation images | `hcis4nwo` |
| 2026-11-08 | DiDiRN-50, 56 px, CP-8, synthetic fixture | Completed one optimizer update and CP validation with epsilon `1e-3`. Wrote a checkpoint. | `jg6sz0zp` |
| 2026-11-08 | DiDiRN-50, 56 px, 25% training data, Weak Aug, CP-8 | One-epoch GPU pilot staged. ImageNet training shards are downloading. | Pending |

## Baseline provenance

The evaluation uses [timm’s public A1 model](https://huggingface.co/timm/resnet50.a1_in1k/tree/767268603ca0cb0bfe326fa87277f19c419566ef) with the pinned DiDiCM validator. The initial run is capped at four CPU cores because other Spark jobs are using the GPU. The run writes final top-1, top-5, and wall time to W&B and stores them in `runs/<run-id>/metrics.json`.

The authors’ repository provides a checkpoint evaluation command. Its public release contains code and training recipes. A paper-trained DiDiRN checkpoint is still needed for a direct checkpoint replication.

## Training data

ImageNet training data is staged from [ILSVRC’s Hugging Face release](https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/49e2ee26f3810fb5a7536bbf732a7b07389a47b5). Validation comes from the same revision. The detached download verifies publisher checksums for all 294 shards. The training pilot requires conversion to ImageFolder and verification of 1,281,167 training and 50,000 validation images.

The CPU smoke fixture used 16 synthetic images for training after the class-balanced subset and 16 for validation. It recorded finite training and validation losses and produced a usable output directory. The fixture exercises the training path and the eight-step sampler. Its accuracy is omitted because the input images were random.
