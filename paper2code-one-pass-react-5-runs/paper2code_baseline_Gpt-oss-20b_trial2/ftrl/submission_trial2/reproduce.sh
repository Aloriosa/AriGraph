#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
apt-get update && apt-get install -y python3-pip
pip install --quiet --upgrade pip
pip install --quiet torch==2.3.0 numpy==1.26.4

# Run the training script
python3 train.py