#!/usr/bin/env bash
set -euo pipefail

# Install system utilities
apt-get update && apt-get install -y python3-pip git

# Upgrade pip
python3 -m pip install --upgrade pip

# Install Python dependencies
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        transformers==4.41.2 datasets==2.19.0 sentence-transformers==3.0.1 tqdm==4.66.2

# Run the main reproduction script
python3 -u src/main.py