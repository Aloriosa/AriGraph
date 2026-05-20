#!/usr/bin/env bash
# reproducibility script
set -e

# Install Python and pip
apt-get update && apt-get install -y python3-pip

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU or GPU automatically selected) and Transformers
pip install torch==2.2.0 transformers==4.38.2

# Run the generation script
python src/generate_with_cfg.py > results.txt

echo "Reproduction completed. Results written to results.txt."