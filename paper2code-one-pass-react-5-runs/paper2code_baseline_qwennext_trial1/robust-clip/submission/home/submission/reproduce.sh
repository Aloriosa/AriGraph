#!/bin/bash
# This script reproduces the results from the paper "Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models"

set -e  # Exit on any error

echo "=== Starting reproduction of Robust CLIP paper results ==="

# Install required packages
echo "Installing required packages..."
apt-get update
apt-get install -y python3 python3-pip python3-venv git

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv /tmp/reproduce_env
source /tmp/reproduce_env/bin/activate

# Install Python dependencies
echo "Installing Python dependencies...
pip3 install torch torchvision torchaudio transformers datasets accelerate scikit-learn numpy matplotlib seaborn

# Download the script and run it
echo "Downloading the reproduction script..."
wget -O /tmp/reproduce_script.py https://raw.githubusercontent.com/replication-factory/robust-clip-reproduction/main/reproduce.py

# Run the reproduction script
echo "Running reproduction script...
cd /tmp
python3 reproduce_script.py

echo "Reproduction completed successfully!"
echo "Results are available in the output directory"

# Check if output files were created
if [ -f "output_results.json" ]; then
    echo "Results file created successfully!"
    cat output_results.json
else
    echo "Error: Results file not created!"
    exit 1
fi