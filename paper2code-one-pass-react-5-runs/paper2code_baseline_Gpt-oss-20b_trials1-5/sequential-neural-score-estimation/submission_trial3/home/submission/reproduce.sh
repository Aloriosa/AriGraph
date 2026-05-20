#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# 1) Install dependencies
# ------------------------------------------------------------
echo "Installing Python packages..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

# ------------------------------------------------------------
# 2) Run the toy experiment
# ------------------------------------------------------------
echo "Running toy SNPSE experiment..."
python3 -m src.main
echo "Done. Results are stored in 'samples.npy' and 'log.txt'."