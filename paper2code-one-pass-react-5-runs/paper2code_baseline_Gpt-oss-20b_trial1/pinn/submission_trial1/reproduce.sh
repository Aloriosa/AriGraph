#!/usr/bin/env bash
set -euo pipefail

# Install required packages
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet torch==2.0.0 numpy scipy

# Run the training script
python3 pinn_convection.py