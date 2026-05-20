#!/usr/bin/env bash
set -euo pipefail

# Install required packages
python -m pip install --upgrade pip
pip install -r requirements.txt

# Sanity check: train NPSE on toy benchmark
python scripts/train_npse.py --benchmark toy --seed 123 --max_iter 500

# Train NPSE and TSNPSE on Gaussian‑Linear benchmark
python scripts/train_npse.py --benchmark gaussian_linear --seed 123 --max_iter 500
python scripts/train_tsnpse.py --benchmark gaussian_linear --seed 123 --max_iter 500

# Sample from the trained Gaussian‑Linear posterior
python scripts/sample.py --model npse_gaussian_linear_5000.pt --n_samples 2000
echo "Reproduction script finished successfully."