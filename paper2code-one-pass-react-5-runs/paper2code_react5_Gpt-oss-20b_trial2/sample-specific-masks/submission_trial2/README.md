# Sample‑specific Masks for Visual Reprogramming (SMM)

This repository contains a faithful implementation of the
*Sample‑specific Masks for Visual Reprogramming* (SMM) method from  
Cai et al., 2024.  
The code reproduces the experiments reported in the paper for a
subset of the 11 datasets (CIFAR‑10/100, SVHN, GTSRB, Flowers‑102,
DTD, UCF‑101, Food‑101, EuroSAT, Oxford‑Pets, SUN‑397).  
Only the datasets that are available through `torchvision`/`torchvision.datasets` are
fully supported; the remaining datasets can be added manually.

## Features

* **Mask generator** – a lightweight 5‑layer CNN producing a 3‑channel mask.
* **Patch‑wise interpolation** – nearest‑neighbour up‑sampling of the mask.
* **Visual re‑programming** – a learnable pattern `δ` multiplied with the mask.
* **Iterative label mapping (Ilm)** – deterministic one‑to‑one mapping from ImageNet classes to target classes.
* **Baselines** – padding, narrow, medium, and full watermarking masks.
* **Backbones** – ImageNet‑pretrained ResNet‑18 and ViT‑B32 (weights frozen).
* **Checkpointing** – best model (including mask generator) is saved.
* **Learning‑rate schedule** – multi‑step scheduler (milestones 100 / 145 epochs).
* **Reproduction script** – `reproduce.sh` trains on a single dataset for a few epochs and prints the test accuracy.

> **Note**: The full experimental protocol (200 epochs, 3‑fold seeds, etc.) is
> available in the original paper. The training script below uses a
> shorter schedule to allow quick verification in a Docker container.

## Usage

```bash
# 1. Install dependencies (PyTorch 2.3.0, torchvision 0.18.0)
pip install --quiet torch torchvision torchaudio

# 2. Run the training script
python src/train_smm.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --baseline smm \
    --epochs 10 \
    --batch-size 256 \
    --lr 0.01 \
    --seed 42
```

The script will download the dataset, train the model, evaluate on the test set,
and print the final test accuracy.  All outputs are deterministic for a fixed
seed.

## Reproduction

The `reproduce.sh` script demonstrates a minimal run that trains the SMM
model for 10 epochs on CIFAR‑10.  It can be invoked as follows:

```bash
bash reproduce.sh
```

The script will install the required packages, run the training script,
and print the test accuracy.

---

> **Caveat**: The implementation focuses on correctness and
> reproducibility rather than raw speed.  The training time may be
> longer than the original paper due to the use of a single GPU
> and a simplified training loop.