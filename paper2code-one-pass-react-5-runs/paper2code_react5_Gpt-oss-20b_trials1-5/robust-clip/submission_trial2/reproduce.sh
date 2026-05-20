#!/usr/bin/env bash
set -euo pipefail

# 1. Install dependencies
echo "=== Installing dependencies ==="
pip install -r requirements.txt

# 2. Download CIFAR‑10 dataset (will be cached in ~/.cache/torchvision)
echo "=== Downloading CIFAR‑10 dataset ==="
python -c "import torchvision; torchvision.datasets.CIFAR10(root='data', train=True, download=True); \
          torchvision.datasets.CIFAR10(root='data', train=False, download=True)"

# 3. Train FARE‑CLIP model
echo "=== Training FARE‑CLIP (2 epochs) ==="
python train_fare.py \
    --batch_size 128 \
    --epochs 2 \
    --lr 1e-5 \
    --wd 1e-4 \
    --epsilon 2/255 \
    --adv_steps 10 \
    --step_size 1/255 \
    --output checkpoints/fare_clip.pt

# 4. Run demo
echo "=== Running demo ==="
python demo.py \
    --clip_model openai/clip-vit-base-patch32 \
    --fare_ckpt checkpoints/fare_clip.pt \
    --output_dir demo

echo "=== Done ==="