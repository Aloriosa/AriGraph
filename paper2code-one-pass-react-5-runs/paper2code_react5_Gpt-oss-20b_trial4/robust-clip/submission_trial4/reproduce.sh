#!/usr/bin/env bash
set -euo pipefail

# ---- 1. Install dependencies ----------------------------------------------
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ---- 2. Download ImageNet (subset) and zero‑shot datasets -------------------
echo "Downloading ImageNet validation set (subset)..."
# We use the public ImageNet validation subset (50k images) shipped with torchvision.
# If you want the full set, replace this with your own ImageNet path.
mkdir -p data/imagenet_val
python download_imagenet_val.py

echo "Downloading CIFAR-10 (for zero‑shot evaluation)..."
python download_cifar10.py

# ---- 3. Train a FARE‑CLIP model ------------------------------------------
echo "Training FARE‑CLIP..."
python train_fare.py \
    --data_dir data/imagenet_val \
    --epochs 2 \
    --batch_size 128 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --epsilon 4/255 \
    --adv_steps 10 \
    --output_dir checkpoints/fare_clip

# ---- 4. Evaluate on ImageNet (clean & robust) ----------------------------
echo "Evaluating on ImageNet (clean)..."
python evaluate.py \
    --model_path checkpoints/fare_clip/model.pt \
    --data_dir data/imagenet_val \
    --split val \
    --eps 0 \
    --output_file results/imagenet_clean.csv

echo "Evaluating on ImageNet (robust, eps=2/255)..."
python evaluate.py \
    --model_path checkpoints/fare_clip/model.pt \
    --data_dir data/imagenet_val \
    --split val \
    --eps 2/255 \
    --output_file results/imagenet_eps2.csv

echo "Evaluating on ImageNet (robust, eps=4/255)..."
python evaluate.py \
    --model_path checkpoints/fare_clip/model.pt \
    --data_dir data/imagenet_val \
    --split val \
    --eps 4/255 \
    --output_file results/imagenet_eps4.csv

# ---- 5. Zero‑shot evaluation on CIFAR‑10 -----------------------------------
echo "Zero‑shot evaluation on CIFAR‑10..."
python zero_shot.py \
    --model_path checkpoints/fare_clip/model.pt \
    --data_dir data/cifar10 \
    --output_file results/cifar10_zeroshot.csv

echo "Reproduction finished. Results are stored in the 'results/' directory."