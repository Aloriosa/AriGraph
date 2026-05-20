#!/usr/bin/env bash
set -euo pipefail

# Update package lists and install python3-pip
apt-get update -qq
apt-get install -y python3-pip

# Install Python dependencies
pip install -r requirements.txt

# Run the training and evaluation script
python train_fare.py