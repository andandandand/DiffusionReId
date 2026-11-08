# Experiment log

| Date (UTC) | Run | Status | W&B ID |
| --- | --- | --- | --- |
| 2026-11-08 | ImageNet-1k ResNet-50 A1, 224 px, 0.95 crop, FP32, Spark CPU | Completed on 50,000 validation images. 80.382% top-1, 94.582% top-5, 7,374.85 s. | `hcis4nwo` |
| 2026-11-08 | DiDiRN-50, 56 px, CP-8, synthetic fixture | Completed one optimizer update and CP validation with epsilon `1e-3`. Wrote a checkpoint. | `jg6sz0zp` |
| 2026-11-08 | DiDiRN-50, 56 px, 25% training data, Weak Aug, CP-8 | Completed one GPU epoch in 21m 17s. 0.096% top-1 and 0.528% top-5 on 50,000 validation images. Saved a checkpoint. | `pno0h2vl` |

## Baseline provenance

The evaluation uses [timm’s public A1 model](https://huggingface.co/timm/resnet50.a1_in1k/tree/767268603ca0cb0bfe326fa87277f19c419566ef) with the pinned DiDiCM validator. The initial run is capped at four CPU cores because other Spark jobs are using the GPU. The run writes final top-1, top-5, and wall time to W&B and stores them in `runs/<run-id>/metrics.json`.

The result matches the published A1 top-1 score and is 0.016 percentage points below its 94.598% top-5 score. Both metrics meet the pilot queue’s 0.20 percentage point tolerance. [timm v1.0.19 reference](https://github.com/huggingface/pytorch-image-models/blob/v1.0.19/results/results-imagenet.csv)

The authors’ repository provides a checkpoint evaluation command. Its public release contains code and training recipes. A paper-trained DiDiRN checkpoint is still needed for a direct checkpoint replication.

## Training data

ImageNet training data is staged from [ILSVRC’s Hugging Face release](https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/49e2ee26f3810fb5a7536bbf732a7b07389a47b5). Validation comes from the same revision. The detached download verifies publisher checksums for all 294 shards and the converter checks coverage of all 1,281,167 training records. If a second full copy would exceed the Spark disk allowance, the compact converter extracts the exact upstream 25% sample and records its order. The launcher checks the manifest and trains on those images with the full 50,000-image validation split.

The compact split was prepared on Spark from all 294 verified shards. It contains 320,255 training images in 1,000 classes and uses about 35 GiB. The ordered manifest’s SHA-256 is `c0ac2ae80616e19e9d873cb2e1607cf3bf9bbe7457c16045f297f3421e78674f`.

## DiDiCM pilot result

The one-epoch run used the verified compact split, FP16 mixed precision, and a diffusion epsilon of `1e-3`. It reported finite training loss of 7.00287 and evaluation loss of 6.90624. The top-1 and top-5 scores were near chance. The checkpoint and final metrics are in `runs/didicm-pilot-imgnet56-20261108T063643Z` on Spark.

A CPU spot check used 16 evenly spaced validation images. The checkpoint's CP-8 sampler returned finite probabilities and two distinct top predictions in both FP32 and FP16 autocast. This check leaves CUDA-specific precision effects open.

The CPU smoke fixture used 16 synthetic images for training after the class-balanced subset and 16 for validation. It recorded finite training and validation losses and produced a usable output directory. The fixture exercises the training path and the eight-step sampler. Its accuracy is omitted because the input images were random.
