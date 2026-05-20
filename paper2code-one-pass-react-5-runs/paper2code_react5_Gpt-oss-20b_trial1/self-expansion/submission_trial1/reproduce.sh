#!/usr/bin/env bash
# reproduce.sh
set -euo pipefail

# Install system packages for building PyTorch (pre‑compiled wheels are used, no need to build)
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3-pip \
    git \
    wget \
    libjpeg-dev \
    libpng-dev

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Download CIFAR‑100 dataset from torchvision
python -c "import torchvision; torchvision.datasets.CIFAR100(root='data', download=True)"

# Run the training & evaluation script
python src/run_sema.py