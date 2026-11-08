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

The pilot uses a diffusion epsilon of `1e-3`. The upstream `1e-5` setting can round the score-entropy ratio to zero at low noise for 1,000 classes, producing a non-finite loss. The pilot keeps recovery checkpoints so the trained weights remain available if validation fails.

### Prepare the training split

Download and verify the 294 training Parquet shards from the [pinned ILSVRC publisher release](https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/49e2ee26f3810fb5a7536bbf732a7b07389a47b5). Keep the validation ImageFolder and its numeric-label-to-synset TSV available. The converter restores original filenames, checks the labels and images, and publishes `train/` when all 1,281,167 images are present.

```bash
dataset="$HOME/Projects/Datasets/Imagenet"
label_map="$(realpath "$dataset/val")/imagenet_label_to_wnid.tsv"
docker run --rm --pull=never --user "$(id -u):$(id -g)" --cpus 2 --memory 4g \
  --mount "type=bind,src=$dataset/train-parquet-49e2ee26,dst=/shards,readonly" \
  --mount "type=bind,src=$dataset,dst=/output" \
  --mount "type=bind,src=$label_map,dst=/labels.tsv,readonly" \
  --mount "type=bind,src=$PWD/scripts/prepare-imagenet.py,dst=/prepare.py,readonly" \
  --entrypoint python diffusion-reid:paper-reproduction /prepare.py \
  /shards /output --label-map /labels.tsv --min-free-gib 180
```

When disk space limits full extraction, run the same converter with `--pilot-subset` and bind a separate directory such as `$dataset/pilot25` to `/output`. The converter uses the pinned timm ordering and upstream seed to write the exact 25% sample and its order to `train-subset-25.txt`. Link the existing validation split at `$dataset/pilot25/val`, then pass `$dataset/pilot25` to the pilot launcher. It checks the manifest and restores the original sample order before training.

## ApexID after ImageNet

Use unseen-animal retrieval as the primary experiment. Train on the supplied training split, use held-out training identities to tune the retrieval protocol, then form disjoint gallery and query sets from the full unknown validation split. Keep near-duplicate groups on one side of the gallery and query partition. Report Recall@1, Recall@5, and mAP, including per-identity results and the number of eligible queries. Report the smaller cross-date cohort separately. Keep the extra unknown split separate because its identities overlap the unknown validation split and its image counts are highly imbalanced.

Use the known-animal validation split for a matched ResNet versus DiDiRN classification comparison. At ApexID’s class count, the upstream probability sampler’s dense class-by-class tensors need substantial memory. Verify an equivalent implementation on small label spaces before attempting full-scale probability sampling.
