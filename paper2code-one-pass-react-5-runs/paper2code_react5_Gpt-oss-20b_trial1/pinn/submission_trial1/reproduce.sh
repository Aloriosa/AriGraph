#!/usr/bin/env bash
set -e

# Ensure reproducible environment
export PYTHONUNBUFFERED=1

# Install dependencies (PyTorch 2.0, numpy, scipy, tqdm)
pip install --quiet --upgrade pip
pip install --quiet torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install --quiet numpy scipy tqdm

# Run the full experiment pipeline
python src/train_pinn.py

echo "Reproduction finished. Results are saved to results.csv."