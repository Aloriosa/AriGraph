#!/usr/bin/env bash
set -euo pipefail

# Update and install system packages
echo "Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3-pip git

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Train the model
echo "Starting training..."
python train.py

# Run sampling
echo "Generating sample..."
python sample.py

echo "Done! Sample image saved to sample.png"