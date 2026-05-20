#!/usr/bin/env bash
set -euo pipefail

# Ensure reproducibility
export PYTHONUNBUFFERED=1

# Install dependencies
pip install -q -r requirements.txt

# Train the adapter
python src/train.py --config config.yaml

# Run inference on the test set
python src/inference.py --config config.yaml

echo "Reproduction finished. See output/predictions.csv for results."