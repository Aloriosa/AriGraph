#!/usr/bin/env bash
set -euo pipefail

# Create a lightweight virtual environment (optional)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the training script
python src/train.py