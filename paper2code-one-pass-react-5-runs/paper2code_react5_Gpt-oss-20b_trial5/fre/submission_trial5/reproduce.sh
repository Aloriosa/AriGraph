#!/usr/bin/env bash
set -e

# Install dependencies
python3 -m pip install --quiet -r requirements.txt

# Create directories for checkpoints
mkdir -p checkpoints policy_checkpoints

# Train FRE encoder/decoder
echo "=== Training FRE ==="
python3 train_fre.py \
    --dataset antmaze-large-diverse-v2 \
    --epochs 10 \
    --steps_per_epoch 500 \
    --output_dir checkpoints

# Train FRE‑conditioned policy (IQL)
echo "=== Training Policy ==="
python3 train_policy.py \
    --dataset antmaze-large-diverse-v2 \
    --fre_checkpoint checkpoints/fre_final.pt \
    --output_dir policy_checkpoints \
    --epochs 5 \
    --steps_per_epoch 500

# Evaluate zero‑shot performance
echo "=== Evaluation ==="
python3 evaluate.py \
    --dataset antmaze-large-diverse-v2 \
    --fre_checkpoint checkpoints/fre_final.pt \
    --policy_checkpoint policy_checkpoints/policy_final.pt \
    --env_id AntMaze-v2 \
    --episodes 10 > evaluation.txt

echo "Reproduction finished. Results in evaluation.txt"