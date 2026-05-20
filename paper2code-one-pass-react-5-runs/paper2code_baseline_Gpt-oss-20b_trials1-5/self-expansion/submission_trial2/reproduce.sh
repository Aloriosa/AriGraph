#!/usr/bin/env bash
set -e

# Update package lists and install pip
apt-get update -qq
apt-get install -y -qq python3-pip

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Download CIFAR‑10 dataset (handled automatically by torchvision)
python3 - <<'PY'
import torchvision
print("Downloading CIFAR‑10 dataset...")
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print("Download finished.")
PY

# Run training
echo "Running SEMA training..."
python3 train_sema.py

echo "Reproduction finished."