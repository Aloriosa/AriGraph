#!/usr/bin/env bash
# reproduce.sh
# This script installs dependencies, trains a small conditional diffusion model
# on CIFAR‑10 using a data‑dependent coupling for in‑painting, and then
# generates a few samples.

set -e

# 1. Install required packages
pip install --quiet torch==2.2.0 torchvision==0.17.0 tqdm numpy pillow

# 2. Train the model
python train.py --epochs 5 --batch 128 --output_dir ./model

# 3. Generate samples
python sample.py --model_path ./model/model.pth --num_samples 10 --output_dir ./samples

echo "Reproduction finished."