#!/usr/bin/env bash
# --------------------------------------------------------------
# Reproduction script for a toy implementation of the Simformer
# as described in the paper "All‑in‑one simulation‑based inference".
#
# The script:
#   1. Installs the required Python packages.
#   2. Runs the training and inference pipeline.
#   3. Generates a CSV file with posterior samples for a toy
#      two‑moons simulation‑based inference task.
# --------------------------------------------------------------

set -euo pipefail

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Running training & inference pipeline ==="
python -u scripts/train_and_eval.py

echo "=== Reproduction complete ==="
echo "Posterior samples are stored in 'posterior_samples.csv'"