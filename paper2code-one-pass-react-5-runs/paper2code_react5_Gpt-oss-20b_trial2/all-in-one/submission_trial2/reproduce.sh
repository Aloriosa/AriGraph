#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------
# 1. Install system packages (minimal)
# -----------------------------------
echo "Updating apt package index..."
sudo apt-get update -qq

echo "Installing Python 3 and pip..."
sudo apt-get install -y python3 python3-pip

# -----------------------------------
# 2. Install Python dependencies
# -----------------------------------
echo "Installing Python dependencies..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# -----------------------------------
# 3. Train the Simformer
# -----------------------------------
echo "Training the Simformer..."
python train.py

# -----------------------------------
# 4. Run inference
# -----------------------------------
echo "Running inference (posterior sampling)..."
python inference.py

echo "Reproduction finished successfully."