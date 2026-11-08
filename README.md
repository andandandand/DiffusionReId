# DiffusionReId

This project first checks the ImageNet results in [Advancing Image Classification with Discrete Diffusion Classification Modeling](https://openaccess.thecvf.com/content/CVPR2026/html/Belhasin_Advancing_Image_Classification_with_Discrete_Diffusion_Classification_Modeling_CVPR_2026_paper.html), then evaluates the approach on ApexID. The [experiment log](EXPERIMENTS.md) records run status and references. The [project board](https://github.com/users/andandandand/projects/16/views/1) tracks pending work.

## Set up on DGX Spark

Use an ARM64 Spark with NVIDIA Container Toolkit. From the repository root, fetch and pin the authors’ code, then build the evaluation image.

```bash
mkdir -p upstream
git clone https://github.com/omer-belhasin/didicm.git upstream/didicm
git -C upstream/didicm checkout --detach 7392557b5e40274a69e797cd910d42b50310f0ac
docker build -t diffusion-reid:paper-reproduction - < Dockerfile
```

Set `WANDB_API_KEY`, `WANDB_BASE_URL=https://api.forge.coreweave.com`, `WANDB_ENTITY`, and `WANDB_PROJECT` in a local `.env`. Set its permissions to `600`. The launchers mount it read-only and Git ignores it.

## ImageNet experiments

Use the ILSVRC 2012 ImageFolder layout with 1,000 synset directories in each split. The baseline can run with the 50,000-image validation split alone. Supply `cpu` to cap the job at four CPU cores when other jobs occupy the GPU.

```bash
scripts/run-baseline.sh ~/Projects/Datasets/Imagenet/val cpu
```

The baseline uses the public ResNet-50 A1 weights and their 224-pixel, 0.95 crop protocol. The paper reports 80.42% top-1 and 94.60% top-5 for its own Strong Aug ResNet-50 at 224 pixels and full training data. The public checkpoint’s published reference is 80.382% and 94.598%. Use the public checkpoint reference when checking the evaluation. [timm v1.0.19 reference](https://github.com/huggingface/pytorch-image-models/blob/v1.0.19/results/results-imagenet.csv)

Run the synthetic test to verify a training update, CP validation, checkpoint writing, and W&B logging. Use the full ImageFolder root for a measured one-epoch DiDiRN-50 pilot.

```bash
scripts/run-didicm-pilot.sh --smoke
scripts/run-didicm-pilot.sh ~/Projects/Datasets/Imagenet
```

The pilot uses 56-pixel inputs, 25% class-balanced training data, the published code’s Weak Aug recipe, and eight CP sampling steps. It uses one GPU with gradient accumulation, a one-epoch schedule, and zero warm-up. Measure wall time and peak memory before selecting a full reproduction budget. The original recipe uses eight GPUs and 600 epochs. For the paper comparison, Table 1 reports 57.92% top-1 and 80.26% top-5 for DiDiRN-50, against 44.86% and 66.46% for the Weak Aug ResNet-50 in that cell. [Authors’ code and recipes](https://github.com/omer-belhasin/didicm/tree/7392557b5e40274a69e797cd910d42b50310f0ac)

## ApexID after ImageNet

Use unseen-animal retrieval as the primary experiment. Train on the supplied training split, use held-out training identities to tune the retrieval protocol, then form disjoint gallery and query sets from the unknown validation identities. Report Recall@1, Recall@5, and mAP, including per-identity results. Keep the extra unknown split separate because its identities overlap the unknown validation split and its image counts are highly imbalanced.

Use the known-animal validation split for a matched ResNet versus DiDiRN classification comparison. At ApexID’s class count, the upstream probability sampler’s dense class-by-class tensors need substantial memory. Verify an equivalent implementation on small label spaces before attempting full-scale probability sampling.
