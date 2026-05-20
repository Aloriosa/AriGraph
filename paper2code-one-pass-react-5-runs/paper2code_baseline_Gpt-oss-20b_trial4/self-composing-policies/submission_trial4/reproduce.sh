#!/usr/bin/env bash
set -euo pipefail

# Update system and install Python3
apt-get update
apt-get install -y python3 python3-pip

# Install PyTorch with CUDA support
pip install --no-cache-dir torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# Install gymnasium
pip install --no-cache-dir gymnasium==0.29.0

# Create results directory
mkdir -p results

# Run training
python src/main.py

echo "All done. Results written to results/results.csv"