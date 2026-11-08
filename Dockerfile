FROM nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21
RUN python -m pip install --no-cache-dir timm==1.0.19 wandb==0.31.0
