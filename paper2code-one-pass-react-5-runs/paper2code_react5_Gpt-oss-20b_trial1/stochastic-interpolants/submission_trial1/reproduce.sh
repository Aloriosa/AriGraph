#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
apt-get update -y
apt-get install -y python3-pip
pip install --upgrade pip
pip install -r requirements.txt

# Create assets directory
mkdir -p assets

# Train in‑painting model
echo "Training in‑painting model..."
python train.py --task inpaint --epochs 10 --batch_size 128 --output_dir assets/inpaint
echo "In‑painting training complete."

# Train super‑resolution model
echo "Training super‑resolution model..."
python train.py --task superres --epochs 10 --batch_size 128 --output_dir assets/superres
echo "Super‑resolution training complete."

# Compute PSNR for both tasks
echo "Evaluating..."
python evaluate.py --input_dir assets/inpaint --output_dir assets/inpaint
python evaluate.py --input_dir assets/superres --output_dir assets/superres

# Consolidate results
echo "PSNR (in‑painting):" > assets/results.txt
cat assets/inpaint/psnr.txt >> assets/results.txt
echo "" >> assets/results.txt
echo "PSNR (super‑resolution):" >> assets/results.txt
cat assets/superres/psnr.txt >> assets/results.txt

echo "Reproduction finished. Results written to assets/results.txt."