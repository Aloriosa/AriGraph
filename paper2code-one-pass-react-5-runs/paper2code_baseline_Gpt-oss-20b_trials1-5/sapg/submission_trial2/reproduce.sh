#!/usr/bin/env bash
set -euo pipefail
# Install required packages
pip install --quiet -U pip
pip install --quiet torch==2.2.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --quiet gymnasium==0.29.1
pip install --quiet numpy tqdm

# Run the training script
python train.py

echo "Reproduction finished. Check results.csv for final rewards."