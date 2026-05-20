#!/usr/bin/env bash
# ------------------------------------------------------------------
# Reproduction script for the PINN wave‑equation experiment.
# --------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

# Install minimal dependencies
pip install --quiet torch==2.0.0+cu118 torchvision==0.15.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install --quiet numpy

# Run training
python src/train_wave.py --width 50 --adam_lr 1e-3 --adam_steps 5000 --lbfgs_steps 2000

echo "Reproduction finished. Results written to output/results.txt"