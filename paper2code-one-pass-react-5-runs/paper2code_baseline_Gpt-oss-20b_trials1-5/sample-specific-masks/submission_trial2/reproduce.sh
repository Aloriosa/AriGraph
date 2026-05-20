#!/usr/bin/env bash
set -e

# Update package list and install basic dependencies
apt-get update -y
apt-get install -y python3-pip git

# Install Python packages
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Run the training script
python3 src/train.py --epochs 5 --batch-size 256