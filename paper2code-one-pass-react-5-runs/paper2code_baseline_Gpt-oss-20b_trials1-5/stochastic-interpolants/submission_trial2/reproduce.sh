#!/usr/bin/env bash
# This script installs dependencies, trains a simple stochastic‑interpolant model on MNIST,
# and generates a few samples.  The whole process takes well under 7 days on a GPU.

set -e

# Install Python and pip
apt-get update -qq
apt-get install -y -qq python3 python3-pip

# Install Python packages
pip install -q -r requirements.txt

# Train the model
python3 train.py --epochs 5 --batch-size 128

# Generate samples
python3 sample.py --num-samples 10 --output-dir samples

echo "Reproduction finished.  Check the 'samples/' directory for generated images."