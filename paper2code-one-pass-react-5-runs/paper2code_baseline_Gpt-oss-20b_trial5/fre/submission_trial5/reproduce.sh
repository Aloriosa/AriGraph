#!/usr/bin/env bash
# reproduce.sh – reproducibility entry point
# This script is executed in a fresh container by the grader.

set -euo pipefail

echo "Installing dependencies..."
# Install PyTorch with CUDA 11.8 (matching the container's GPU)
pip install --quiet torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
pip install --quiet gymnasium==0.28.1 numpy tqdm

echo "Generating offline dataset..."
python generate_offline_data.py

echo "Training FRE encoder and Q‑policy..."
python train_fre.py

echo "Running zero‑shot evaluation..."
python evaluate.py

echo "Reproduction complete. See results.txt for the final score."