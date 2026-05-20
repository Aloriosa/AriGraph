#!/usr/bin/env bash
set -e

# ------------------------------------------------------------- #
# Install system dependencies
# ------------------------------------------------------------- #
apt-get update && apt-get install -y python3 python3-pip git

# ------------------------------------------------------------- #
# Create a virtual environment (optional but recommended)
# ------------------------------------------------------------- #
python3 -m venv venv
source venv/bin/activate

# ------------------------------------------------------------- #
# Install Python packages
# ------------------------------------------------------------- #
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 diffusers==0.27.2 \
          transformers==4.38.2 accelerate==0.25.0 datasets==2.16.1 \
          torchmetrics==0.11.4 lpips==0.1.4 tqdm

# ------------------------------------------------------------- #
# Run training
# ------------------------------------------------------------- #
echo "=== Training ==="
python train.py

# ------------------------------------------------------------- #
# Run evaluation
# ------------------------------------------------------------- #
echo "=== Evaluation ==="
python evaluate.py

echo "Reproduction finished. Check checkpoints/ and samples/ for results."