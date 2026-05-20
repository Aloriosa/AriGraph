#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------
# 1. Install Python dependencies
# --------------------------------------------------------------
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# --------------------------------------------------------------
# 2. Run the FOA experiment
# --------------------------------------------------------------
echo "Running FOA experiment..."
python3 main.py --batch_size 8 --prompt_size 3 --population 6 --lambda 0.4

echo "Reproduction finished."