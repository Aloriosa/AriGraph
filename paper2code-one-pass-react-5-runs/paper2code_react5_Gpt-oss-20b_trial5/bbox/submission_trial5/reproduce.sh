#!/usr/bin/env bash
# -------------------------------------------------------------
# Reproduction script for the simplified BBox‑Adapter demo
# -------------------------------------------------------------

set -euo pipefail

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch transformers datasets tqdm

# Train the adapter and evaluate
echo "Training adapter..."
python train_adapter.py \
    --train_file dataset/train.jsonl \
    --test_file dataset/test.jsonl \
    --output_dir outputs \
    --epochs 5 \
    --lr 1e-4 \
    --batch_size 1 \
    --num_candidates 5

echo "Reproduction finished."
echo "Results are in the 'outputs/' directory."