#!/usr/bin/env bash
set -e

# Update package list and install system dependencies
apt-get update -y
apt-get install -y python3 python3-pip

# Install Python packages
pip install --upgrade pip
pip install gymnasium==0.29.0 torch==2.1.0 numpy==1.26.0 tqdm==4.66.1

# Run the training script
python sapg.py

# Print a short confirmation
echo "Training complete. Check results.txt for evaluation statistics."