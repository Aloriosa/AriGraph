#!/usr/bin/env bash
set -euo pipefail

# Install the minimal dependency set
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run the training script
echo "Starting SAPG training..."
python train.py

echo "Training finished. Results are in the 'logs' directory."