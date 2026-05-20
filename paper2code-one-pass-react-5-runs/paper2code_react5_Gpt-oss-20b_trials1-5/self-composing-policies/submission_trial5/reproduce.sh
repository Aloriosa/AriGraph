#!/usr/bin/env bash
set -euo pipefail

# Install Python packages
pip install -r requirements.txt

# Run the training script
python src/train_componet.py