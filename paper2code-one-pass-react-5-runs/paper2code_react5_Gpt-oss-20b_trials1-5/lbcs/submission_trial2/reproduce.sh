#!/usr/bin/env bash
set -euo pipefail

# Install required packages
echo "Installing Python dependencies..."
apt-get update && apt-get install -y python3-pip
pip3 install --upgrade pip
pip3 install torch torchvision tqdm

# Create output directory
mkdir -p output

# Run the LBCS script on Fashion‑MNIST
echo "Running LBCS on Fashion‑MNIST..."
python3 lbc_solve.py \
    --dataset fmnist \
    --k 200 \
    --epsilon 0.2 \
    --outer-iterations 200 \
    --inner-epochs 3 \
    --batch-size 64 \
    --lr 0.001 \
    --output-dir output

# Show the results
echo "Reproduction finished. Results:"
cat output/results.txt