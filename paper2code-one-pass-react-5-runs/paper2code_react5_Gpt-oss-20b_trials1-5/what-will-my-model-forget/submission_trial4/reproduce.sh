#!/usr/bin/env bash
set -euo pipefail

# Install required Python packages
python -m pip install --upgrade pip
pip install -q torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    transformers==4.42.0 datasets==2.20.0 scikit-learn==1.5.0 tqdm==4.66.5

# Run the experiment
python src/main.py