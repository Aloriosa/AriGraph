#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------
# 1. Install dependencies
# -------------------------------------------------
echo "Installing Python dependencies..."
pip install -q -U pip
pip install -q -r requirements.txt

# -------------------------------------------------
# 2. Download CIFAR‑10 dataset (source domain)
# -------------------------------------------------
echo "Downloading CIFAR‑10 dataset..."
python -c "
import torchvision
torchvision.datasets.CIFAR10(root='data', train=True, download=True)
"

# -------------------------------------------------
# 3. Create target dataset (10 images)
# -------------------------------------------------
echo "Creating 10‑shot target dataset..."
python -c "
import os
import random
from torchvision import datasets, transforms
from PIL import Image

DATA_ROOT = 'data/cifar10'
TARGET_DIR = 'dataset/targets'
os.makedirs(TARGET_DIR, exist_ok=True)

# Load CIFAR‑10
dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False)

# Pick first 10 images as the target
for idx in range(10):
    img, _ = dataset[idx]
    img.save(os.path.join(TARGET_DIR, f'target_{idx}.png'))
"
# -------------------------------------------------
# 4. Train the model (similarity‑guided + adversarial noise)
# -------------------------------------------------
echo "Training the fine‑tuned diffusion model..."
python train.py

# -------------------------------------------------
# 5. Generate images with the fine‑tuned model
# -------------------------------------------------
echo "Generating samples..."
python generate.py

echo "Reproduction finished. Generated images are in ./output/"