#!/usr/bin/env bash
# Reproduction script for the Sequential Neural Posterior Score Estimation (TSNPSE) toy implementation.
# This script installs dependencies, runs the training, and generates posterior samples.

set -euo pipefail

# Create a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch numpy tqdm matplotlib

# Ensure data directory exists
mkdir -p data

# If the toy observed data file does not exist, generate a simple example
if [ ! -f data/observed_data.npy ]; then
    echo "Generating toy observed data (2D Gaussian mixture)..."
    python - <<'PY'
import numpy as np
# Simple 2D mixture: two Gaussian modes
np.random.seed(0)
mode1 = np.random.randn(2) * 0.5 + np.array([1.0, 1.0])
mode2 = np.random.randn(2) * 0.5 + np.array([-1.0, -1.0])
# Randomly pick one mode
x_obs = mode1 if np.random.rand() < 0.5 else mode2
np.save('data/observed_data.npy', x_obs)
print("Saved observed data:", x_obs)
PY
fi

# Run the training script
python -m src.train

echo "Reproduction finished. Posterior samples are saved in outputs/posterior_samples.npy"