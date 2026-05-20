#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install --upgrade pip
pip install torch torchvision transformers datasets tqdm

# Run the reproduction script
python src/finetune_and_evaluate.py