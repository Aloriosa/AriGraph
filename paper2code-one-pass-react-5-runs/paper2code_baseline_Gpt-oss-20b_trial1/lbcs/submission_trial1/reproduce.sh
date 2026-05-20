#!/usr/bin/bash
set -e

# Install dependencies
pip install --quiet -U pip
pip install --quiet -U torch torchvision numpy tqdm

# Run the reproduction script
python -m src.main