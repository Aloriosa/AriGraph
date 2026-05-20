#!/usr/bin/env bash
# reproduce.sh
# Installs required packages, trains the Simformer on the Gaussian‑linear toy task
# and prints a simple posterior evaluation.

set -euo pipefail

# Install PyTorch (CPU+CUDA if available)
pip install --upgrade pip
pip install torch torchvision torchaudio tqdm

# Run training and sampling script
python train.py