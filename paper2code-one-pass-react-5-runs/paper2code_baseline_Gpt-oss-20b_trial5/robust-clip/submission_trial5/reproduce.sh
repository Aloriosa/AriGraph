#!/usr/bin/env bash
set -euo pipefail

# Install system packages
apt-get update
apt-get install -y python3 python3-pip

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create output directory
mkdir -p models

# 1. Fine‑tune CLIP with FARE
echo "=== Training FARE model ==="
python src/finetune.py --epochs 2 --batch 32 --eps 4 --pgd_steps 10

# 2. Evaluate zero‑shot accuracy
echo "=== Zero‑shot evaluation ==="
python src/eval_zs.py