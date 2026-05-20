#!/usr/bin/env bash
# -------------------------------------------------------------
# Reproduction script for the SMM (Sample‑specific Multi‑channel
# Masks) visual reprogramming method.
#
# This script installs the required Python packages, downloads the
# CIFAR‑10 dataset, trains a ResNet‑18 based SMM model for 10
# epochs, evaluates it on the test set and prints the test accuracy.
#
# Usage:
#   bash reproduce.sh
#
# The script is written to be run in a clean Ubuntu 24.04 Docker
# container with an NVIDIA GPU (CUDA 12.x).  It uses PyTorch
# 2.3.0 and torchvision 0.18.0.
# -------------------------------------------------------------

set -e

# 1. Install dependencies
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

# 3. Print a summary of the run
echo
echo "=== Reproduction finished ==="
echo "Test accuracy reported above."