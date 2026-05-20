#!/usr/bin/env bash
set -euo pipefail

# Install system packages (Python, pip)
if ! command -v python3 >/dev/null 2>&1; then
    apt-get update && apt-get install -y python3 python3-pip
fi

# Install Python dependencies
pip install --upgrade pip
pip install torch==2.2.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate tqdm scikit-learn

# Run the reproduction script
python3 apt_finetune.py