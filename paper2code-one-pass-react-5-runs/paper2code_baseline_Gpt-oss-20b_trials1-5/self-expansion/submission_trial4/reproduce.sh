#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing dependencies ==="
# Use the system python (Ubuntu 24.04 has Python 3.12)
python3 -m pip install --upgrade pip
pip install --upgrade setuptools wheel
# Core libraries
pip install torch torchvision timm
# Optional: tqdm for progress bars
pip install tqdm

echo "=== Running training ==="
python3 train_sema.py

echo "=== Training finished ==="
echo "Results are stored in results.txt"