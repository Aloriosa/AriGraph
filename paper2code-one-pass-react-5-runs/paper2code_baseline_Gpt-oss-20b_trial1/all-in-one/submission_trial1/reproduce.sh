#!/usr/bin/env bash
set -euo pipefail

echo "=== Reproduction script for a toy Two Moons SBI experiment ==="

# 1. Install Python dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 2. Train the posterior model
echo "Training posterior model..."
python train.py

# 3. Sample from the posterior
echo "Sampling posterior..."
python sample.py

echo "=== Reproduction finished ==="