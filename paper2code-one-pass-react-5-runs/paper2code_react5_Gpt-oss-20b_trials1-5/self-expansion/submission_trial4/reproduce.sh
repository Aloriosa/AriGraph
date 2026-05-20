#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install -q torch torchvision timm tqdm numpy

# Run training
python -u main.py