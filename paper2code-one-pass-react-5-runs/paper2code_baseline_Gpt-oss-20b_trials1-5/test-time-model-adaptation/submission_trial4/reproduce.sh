#!/usr/bin/env bash
set -e

# Update system and install build-essential for compiling any libraries
apt-get update -y
apt-get install -y --no-install-recommends build-essential

# Install Python 3 and pip
apt-get install -y python3 python3-pip

# Install required Python packages
pip3 install --no-cache-dir numpy torch torchvision timm pycma tqdm

# Run the reproduction script
python3 run_foo.py