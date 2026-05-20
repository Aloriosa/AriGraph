#!/usr/bin/env bash
set -euo pipefail

# Install system packages (only python3 is needed for this toy repo)
apt-get update && apt-get install -y python3

# Install python dependencies
pip install -r requirements.txt

# Create directories for checkpoints and samples
mkdir -p checkpoints samples

# Train the model
python train.py --config config.yaml

# Generate samples
python sample.py --config config.yaml

# Compute FID (optional, may take a while on large datasets)
# torch-fidelity will automatically download a reference set if not provided.
# For the sake of this reproduction we keep it lightweight.
python evaluate_fid.py --config config.yaml