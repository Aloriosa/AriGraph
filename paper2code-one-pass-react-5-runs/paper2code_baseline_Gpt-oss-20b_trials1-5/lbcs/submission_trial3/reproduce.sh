#!/usr/bin/env bash
# Reproduction script for the Refined Coreset Selection demo

set -euo pipefail

echo "Installing required packages..."
pip install --quiet torch torchvision

echo "Running the core script..."
python lbc_s.py

echo "Reproduction finished. Check results.txt for details."