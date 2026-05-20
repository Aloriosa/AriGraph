#!/usr/bin/env bash
set -e

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Prepare data
python -c "import data.target as d; d.download_and_prepare(num_samples=10)"

# Train and evaluate
python train_ants.py \
    --batch_size 8 \
    --clf_epochs 5 \
    --clf_lr 1e-3 \
    --unet_epochs 3 \
    --unet_lr 5e-5 \
    --gamma 5.0 \
    --gen_batch 10 \
    --num_samples 1000

echo "Reproduction finished. Check metrics.txt for results."