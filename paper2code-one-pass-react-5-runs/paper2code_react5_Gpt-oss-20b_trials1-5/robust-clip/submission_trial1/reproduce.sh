#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------------------- #
# 1. Install dependencies
# --------------------------------------------------------------------------- #
echo "Installing Python dependencies..."
pip install -q -r requirements.txt

# --------------------------------------------------------------------------- #
# 2. Train the FARE model
# --------------------------------------------------------------------------- #
echo "Training FARE‑CLIP (unsupervised adversarial fine‑tuning)..."
python train_fare.py \
    --batch-size 32 \
    --epochs 2 \
    --lr 1e-5 \
    --wd 1e-4 \
    --eps 2/255.0 \
    --pgd-steps 10 \
    --step-size 1/255.0 \
    --out-dir checkpoints

# --------------------------------------------------------------------------- #
# 3. Evaluate on clean and adversarial ImageNet validation images
# --------------------------------------------------------------------------- #
echo "Evaluating clean and robust accuracy..."
python evaluate_fare.py \
    --ckpt checkpoints/clip_fare.pt \
    --batch-size 32 \
    --out-dir results

# --------------------------------------------------------------------------- #
# 4. Placeholder LVLM evaluation
# --------------------------------------------------------------------------- #
echo "Running placeholder LVLM evaluation (no heavy computation)..."
python evaluate_lvml.py

echo "Reproduction finished. Results are in the 'results' and 'lvml_results' directories."