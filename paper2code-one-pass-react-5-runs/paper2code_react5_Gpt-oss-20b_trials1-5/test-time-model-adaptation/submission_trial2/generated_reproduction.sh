#!/usr/bin/env bash
# --------------------------------------------------------------
# 1. Install Python dependencies
# --------------------------------------------------------------
set -euo pipefail

python -m pip install --upgrade pip
pip install -q torch torchvision timm numpy pycma tqdm pillow requests

# --------------------------------------------------------------
# 2. Download ImageNet‑C (severity level 5)
# --------------------------------------------------------------
IMAGENETC_DIR="imagenet_c"
if [ ! -d "$IMAGENETC_DIR" ]; then
    echo "Downloading ImageNet‑C (severity level 5)..."
    wget -q -O imagenet_c.tar.gz https://s3.amazonaws.com/imagenet_c/imagenet_c.tar.gz
    mkdir -p "$IMAGENETC_DIR"
    tar -xzf imagenet_c.tar.gz -C "$IMAGENETC_DIR"
    rm imagenet_c.tar.gz
fi

# --------------------------------------------------------------
# 3. Download ImageNet validation images and ground‑truth
# --------------------------------------------------------------
VAL_DIR="ILSVRC2012_img_val"
GT_FILE="ILSVRC2012_validation_ground_truth.txt"

if [ ! -f "$GT_FILE" ]; then
    echo "Downloading ImageNet‑validation ground‑truth..."
    wget -q -O "$GT_FILE" https://s3.amazonaws.com/imagenet_c/ILSVRC2012_validation_ground_truth.txt
fi

if [ ! -d "$VAL_DIR" ]; then
    echo "Downloading ImageNet‑validation images..."
    wget -q -O ILSVRC2012_img_val.tar https://s3.amazonaws.com/imagenet_c/ILSVRC2012_img_val.tar
    mkdir -p "$VAL_DIR"
    tar -xf ILSVRC2012_img_val.tar -C "$VAL_DIR"
    rm ILSVRC2012_img_val.tar
fi

# --------------------------------------------------------------
# 4. Run the FOA evaluation
# --------------------------------------------------------------
echo "Running FOA..."
python generated_reproduction.py \
    --batch-size 64 \
    --prompt-len 3 \
    --popsize 28 \
    --lambda 0.4 \
    --gamma 1.0 \
    --alpha 0.1 \
    --quantize 8 \
    --device cuda \
    --seed 42

echo "Results written to results.txt"