#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
echo "Installing Python dependencies..."
python3 -m pip install --quiet -r requirements.txt

# Run the training script
echo "Running training pipeline..."
python3 train.py

echo "Reproduction finished. Results stored in results.txt"