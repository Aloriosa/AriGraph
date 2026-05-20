#!/usr/bin/env bash
set -euo pipefail

# Install minimal dependencies
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

# Train the FRE encoder/decoder
python3 train_fre.py \
  --num_states 2000 \
  --state_dim 2 \
  --batch_size 32 \
  --K 32 \
  --Kp 8 \
  --epochs 20 \
  --lr 1e-3 \
  --beta 0.01 \
  --out_dir results

# Evaluate on a new reward function
python3 evaluate_fre.py \
  --model_path results/fre_model.pt \
  --num_states 2000 \
  --state_dim 2 \
  --K 32 \
  --out_dir results

echo "Reproduction finished. Results in the 'results/' directory."