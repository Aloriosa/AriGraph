#!/usr/bin/env bash
# Reproduce script for the SMM example
# -------------------------------------------------------------
# 1. Install dependencies (PyTorch + torchvision)
# -------------------------------------------------------------
# The container already has CUDA drivers; we install the matching PyTorch
# wheel for CUDA 12.0 (the default in Ubuntu 24.04).
pip install --quiet torch==2.3.0+cu120 torchvision==0.18.0+cu120 -f https://download.pytorch.org/whl/torch_stable.html

# -------------------------------------------------------------
# 2. Run the training script
# -------------------------------------------------------------
python main.py --epochs 10 --batch-size 128 --lr 0.01

# -------------------------------------------------------------
# 3. Show the final result
# -------------------------------------------------------------
echo "=== Final test accuracy ==="
cat output/results.txt