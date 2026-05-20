#!/usr/bin/env bash
set -euo pipefail

# Install dependencies (will create a lightweight environment)
echo "Installing Python dependencies..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir transformers datasets torch tqdm scikit-learn

# Run the reproduction script
echo "Running the reproduction pipeline..."
python3 main.py