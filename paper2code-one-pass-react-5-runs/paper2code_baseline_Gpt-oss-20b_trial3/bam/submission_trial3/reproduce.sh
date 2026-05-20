#!/usr/bin/env bash
set -euo pipefail

# Create a clean environment
echo "Installing dependencies..."
pip install --quiet -U pip
pip install --quiet -U jax jaxlib==0.4.2+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --quiet -U optax numpy

# Run the synthetic Gaussian experiments
echo "Running Gaussian experiments..."
mkdir -p results
python experiments/run_gaussian.py

echo "Reproduction finished. Results are in the 'results/' directory."