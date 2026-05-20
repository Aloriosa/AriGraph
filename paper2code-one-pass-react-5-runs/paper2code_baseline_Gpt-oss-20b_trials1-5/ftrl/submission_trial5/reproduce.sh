#!/usr/bin/env bash
# ------------------------------------------------------------
# Reproduce script for the toy RL forgetting experiment
# ------------------------------------------------------------
set -euo pipefail

# Install dependencies
pip install -q torch==2.0.0 numpy==1.26.0

# Run the training script
python train.py

# Done
echo "Reproduction finished. Results stored in results.csv"