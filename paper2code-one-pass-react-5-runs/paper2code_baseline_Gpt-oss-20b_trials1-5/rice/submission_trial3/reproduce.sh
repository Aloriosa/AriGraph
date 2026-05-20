#!/usr/bin/env bash
set -euo pipefail

# Install system dependencies
apt-get update
apt-get install -y python3-pip git

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Train baseline PPO agent
python src/train_pretrained.py

# Extract critical states
python src/train_critical.py

# Refine with mixed initial distribution + RND
python src/refine.py

echo "Reproduction finished."