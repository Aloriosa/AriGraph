#!/usr/bin/env bash
set -e

# Install dependencies
echo "Installing dependencies..."
pip install --quiet -q -r requirements.txt

# Run training
echo "Running training script..."
python train_apt.py

echo "Training completed. Results written to output.json."