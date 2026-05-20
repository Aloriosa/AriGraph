#!/usr/bin/env bash
set -euo pipefail

# Install system packages
apt-get update && apt-get install -y python3-pip

# Install Python dependencies
pip install -r requirements.txt

# Run the training script
python3 train_sapg.py