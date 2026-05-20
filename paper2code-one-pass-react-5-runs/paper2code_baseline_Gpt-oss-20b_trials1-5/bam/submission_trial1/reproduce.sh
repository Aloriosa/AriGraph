#!/usr/bin/env bash
set -euo pipefail

# Ensure reproducibility
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export FORCE_CPU=1   # comment out if you want to use GPU (requires jax)

# Install dependencies
pip install --quiet -r requirements.txt

# Create results directory
mkdir -p results

# Run toy Gaussian experiment
python experiments/run_gaussian.py --output-dir results

echo "Reproduction finished. See results/kl_convergence.png for the convergence plot."