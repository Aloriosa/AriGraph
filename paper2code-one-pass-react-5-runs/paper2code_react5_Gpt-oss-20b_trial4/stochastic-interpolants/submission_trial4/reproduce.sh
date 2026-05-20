#!/usr/bin/env bash
set -euo pipefail

# 1. Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 2. Download CIFAR‑10 (handled automatically by torchvision)
python -c "from torchvision import datasets; datasets.CIFAR10(root='data', train=True, download=True); datasets.CIFAR10(root='data', train=False, download=True)"

# 3. Train the model
echo "Training velocity model..."
python train.py --epochs 10 --batch_size 128 --lr 2e-4

# 4. Generate samples
echo "Generating samples..."
python sample.py --task inpainting
python sample.py --task superresolution

# 5. Evaluate
echo "Evaluating FID..."
python eval.py > outputs/eval.txt

echo "Reproduction complete. Check outputs/ directory."