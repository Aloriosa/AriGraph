#!/usr/bin/env bash
set -e

echo "=== Reproduce Stochastic Interpolants (MNIST in‑painting) ==="

# 1. Update package lists and install system dependencies
apt-get update -y
apt-get install -y python3 python3-pip git

# 2. Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# 3. Train the model
echo "Training the velocity model..."
python train_inpainting.py --epochs 5 --batch-size 128 --output model.pt

# 4. Generate samples
echo "Generating samples..."
python sample_inpainting.py --model model.pt --output samples/

# 5. Summary
echo "=== Reproduction finished ==="