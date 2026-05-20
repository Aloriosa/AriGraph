#!/usr/bin/env bash
# Reproduce the PINN wave equation experiment
set -euo pipefail

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet torch==2.0.0 torchvision==0.15.1

# Run the training script
echo "Running training..."
python3 pinn_wave.py