#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------
# 1. Install dependencies
# --------------------------------------------------------------
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# --------------------------------------------------------------
# 2. Run the full experiment pipeline
# --------------------------------------------------------------
echo "Running TSNPSE experiments..."
python3 main.py --output-dir ./output --rounds 5 --sims-per-round 5000 --epochs 15 --baseline-epochs 15
echo "All experiments completed. Results are in the 'output/' directory."