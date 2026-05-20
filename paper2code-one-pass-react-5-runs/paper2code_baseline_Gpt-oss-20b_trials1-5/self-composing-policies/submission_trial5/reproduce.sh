#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
echo "Installing required Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Run training
echo "Starting training..."
python train.py