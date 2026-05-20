#!/usr/bin/env bash
# reproducible reproduction script
# --------------------------------------------------
# This script installs the minimal dependencies and runs a
# short Python program that demonstrates the environment.
# --------------------------------------------------
set -euo pipefail

# Install Python packages
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -q torch torchvision

# Run the reproduction script
echo "Running the reproduction script..."
python -m src.reproduce

# End of script
echo "Reproduction finished successfully."