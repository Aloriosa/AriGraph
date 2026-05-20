#!/usr/bin/env bash
# Reproduce script for Sample‑specific Masks for Visual Reprogramming (SMM)
# This script installs dependencies, downloads datasets, trains a SMM model
# on CIFAR‑10 with a ResNet‑18 backbone, and prints the test accuracy.

set -e

# 1. Install system dependencies (non‑interactive)
apt-get update && apt-get install -y python3-pip git

# 2. Install Python packages
pip install --upgrade pip
pip install torch torchvision timm tqdm numpy pillow

# 3. Run the training script
python src/main.py \
  --dataset cifar10 \
  --backbone resnet18 \
  --epochs 10 \
  --batch-size 256 \
  --lr 0.01 \
  --lr-decay 0.1 \
  --seed 42

echo "Reproduction finished. Check stdout for final test accuracy."