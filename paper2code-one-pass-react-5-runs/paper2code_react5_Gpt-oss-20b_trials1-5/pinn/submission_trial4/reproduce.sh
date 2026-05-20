#!/usr/bin/env bash
set -euo pipefail

# Install minimal dependencies
pip install -q -U torch==2.0.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q -U numpy

# Create results directory
mkdir -p results

# Run the full experiment
python src/pinn_wave.py --output results/results.csv