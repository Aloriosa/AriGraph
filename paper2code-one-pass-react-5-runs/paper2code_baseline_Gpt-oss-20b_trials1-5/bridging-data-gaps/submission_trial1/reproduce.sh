#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# 1. Install dependencies
# ------------------------------------------------------------------
echo "Installing dependencies..."
pip install -q -U pip
pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q diffusers==0.21.3 transformers datasets accelerate tqdm

# ------------------------------------------------------------------
# 2. Prepare output directories
# ------------------------------------------------------------------
mkdir -p output/generated
rm -rf output/model.pt
rm -f output/metrics.txt

# ------------------------------------------------------------------
# 3. Fine‑tune the diffusion model
# ------------------------------------------------------------------
echo "Fine‑tuning the diffusion model on 10 CIFAR‑10 images..."
python src/train.py \
    --dataset cifar10 \
    --num_samples 10 \
    --epochs 3 \
    --batch_size 4 \
    --lr 5e-5 \
    --output_dir output

# ------------------------------------------------------------------
# 4. Generate images from the adapted model
# ------------------------------------------------------------------
echo "Generating images from the fine‑tuned model..."
python src/generate.py \
    --model_path output/model.pt \
    --num_images 5 \
    --output_dir output/generated

echo "Reproduction completed. Generated images in output/generated/"