#!/usr/bin/env bash
set -euo pipefail

# Install python packages
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Prepare directories
mkdir -p results

# Train and sample Gaussian‑Linear task
echo "=== Gaussian‑Linear ==="
python3 -m src.train --task gaussian_linear --epochs 50 --batch_size 512 > logs/gaussian_linear.log 2>&1

echo "=== Sampling Gaussian‑Linear ==="
python3 -m src.sample --task gaussian_linear --n_samples 2000 > logs/gaussian_linear_sample.log 2>&1

# Train and sample Two‑Moons task
echo "=== Two‑Moons ==="
python3 -m src.train --task two_moons --epochs 50 --batch_size 512 > logs/two_moons.log 2>&1

echo "=== Sampling Two‑Moons ==="
python3 -m src.sample --task two_moons --n_samples 2000 > logs/two_moons_sample.log 2>&1

echo "=== Done ==="