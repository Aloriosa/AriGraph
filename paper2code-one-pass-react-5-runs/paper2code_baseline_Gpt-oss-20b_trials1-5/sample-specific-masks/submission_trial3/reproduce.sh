#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------
# Reproduction script for SMM
# -------------------------------------------------------------
# 1. Install system dependencies
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git

# 2. Create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate

# 3. Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run a short training experiment
#    (you can modify arguments as needed)
python train_smm.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --epochs 10 \
    --batch_size 256 \
    --lr 1e-3 \
    --patch_size 8

# 5. Deactivate the virtual environment
deactivate