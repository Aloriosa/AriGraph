#!/usr/bin/env bash
set -e

# Update package lists and install Python 3 pip
apt-get update -y
apt-get install -y python3 python3-pip

# Install Python dependencies
pip3 install --upgrade pip
pip3 install torch torchvision timm tqdm numpy

# Run the training and evaluation script
python3 train_sema.py