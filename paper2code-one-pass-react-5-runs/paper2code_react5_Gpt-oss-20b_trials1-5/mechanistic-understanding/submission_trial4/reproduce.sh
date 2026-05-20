#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
apt-get update
apt-get install -y python3 python3-pip git
pip install --upgrade pip
pip install -r requirements.txt

# Train probe
python train_probe.py

# Extract toxic vectors
python extract_toxic_vectors.py

# Generate pairwise data
python pairwise_dataset.py

# Train DPO
python train_dpo.py

# Evaluate
python evaluate.py

echo "Reproduction completed. Results are in the output directories."