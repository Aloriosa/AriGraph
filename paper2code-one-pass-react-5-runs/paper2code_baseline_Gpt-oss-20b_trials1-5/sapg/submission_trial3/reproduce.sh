#!/usr/bin/env bash
set -e

# Update system and install Python 3
apt-get update -qq
apt-get install -y python3 python3-pip ffmpeg

# Install Python dependencies
pip install --no-cache-dir torch==2.3.0+cpu torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir gymnasium==0.30.1 numpy tqdm

# Run the training script with default hyper‑parameters
python3 train_sapg.py > training.log 2>&1

echo "Training finished. See training.log for details."