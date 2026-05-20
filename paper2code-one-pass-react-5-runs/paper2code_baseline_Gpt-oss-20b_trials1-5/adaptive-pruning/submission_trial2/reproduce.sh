#!/usr/bin/env bash
set -euo pipefail

# Install system dependencies
apt-get update && apt-get install -y python3 python3-pip

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Run the training script
python train.py