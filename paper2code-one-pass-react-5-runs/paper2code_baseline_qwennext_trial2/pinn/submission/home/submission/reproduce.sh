#!/bin/bash

# Set up the environment
set -e  # Exit on any error

echo "Setting up environment for PINN reproduction..."

# Install required packages
apt-get update
apt-get install -y python3 python3-pip python3-venv git

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install torch torchvision torchaudio numpy matplotlib scikit-learn

# Create necessary directories
mkdir -p /home/submission/models
mkdir -p /home/submission/results

echo "Environment setup complete."

# Run the PINN reproduction script
echo "Running PINN reproduction script..."
python3 pinns_reproduction.py

echo "Reproduction complete!"