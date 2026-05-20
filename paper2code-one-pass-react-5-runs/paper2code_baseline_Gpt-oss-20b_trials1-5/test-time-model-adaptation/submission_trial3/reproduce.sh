#!/usr/bin/env bash
# This script sets up the environment and runs the FOA reproduction
# It is designed to work on an Ubuntu 24.04 LTS Docker container
# with an NVIDIA GPU (via the container toolkit).

set -e

# Install Python 3 and pip
apt-get update
apt-get install -y python3-pip

# Install required Python packages
pip install --no-cache-dir torch torchvision timm pycma tqdm

# Run the reproduction script
python3 main.py