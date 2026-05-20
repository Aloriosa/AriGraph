#!/usr/bin/env bash
set -e

# Update package list and install Python 3 + pip
apt-get update -y
apt-get install -y python3 python3-pip

# Upgrade pip and install required Python packages
python3 -m pip install --upgrade pip
python3 -m pip install torch==2.1.0 transformers==4.40.0 datasets==2.20.0 tqdm==4.66.2

# Run the reproduction script
python3 main.py