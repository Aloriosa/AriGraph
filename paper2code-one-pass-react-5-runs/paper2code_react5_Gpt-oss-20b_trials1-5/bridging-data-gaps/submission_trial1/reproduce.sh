#!/usr/bin/env bash
set -euo pipefail

# Install system packages
apt-get update && apt-get install -y git

# Create a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the training and evaluation pipeline
python train.py