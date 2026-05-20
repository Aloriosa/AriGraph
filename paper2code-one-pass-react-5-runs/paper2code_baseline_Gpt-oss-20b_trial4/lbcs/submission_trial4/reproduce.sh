#!/usr/bin/env bash
set -e

# Update packages and install Python 3.10 and pip
apt-get update
apt-get install -y python3.10 python3-pip

# Install required Python packages
pip install --no-cache-dir torch torchvision tqdm

# Run the main script
python3 main.py