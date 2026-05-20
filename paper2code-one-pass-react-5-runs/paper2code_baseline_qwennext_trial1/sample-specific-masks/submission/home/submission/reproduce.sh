#!/bin/bash
# This script reproduces the results from the paper:
# "Sample-specific Masks for Visual Reprogramming-based Prompting"

# Set up environment
echo "Setting up environment..."
apt-get update
apt-get install -y python3 python3-pip git

# Install required packages
echo "Installing required packages..."
pip3 install torch torchvision numpy matplotlib scikit-learn tqdm

# Clone the repository (if needed) - but we'll use local files
# Create necessary directories
mkdir -p /home/submission/data
mkdir -p /home/submission/models
mkdir -p /home/submission/results

# Download datasets if not available
echo "Downloading datasets (if needed)..."
# Note: In a real implementation, we'd download datasets, but for reproduction, 
# we'll use torchvision's built-in dataset downloading

# Copy all source files from the submission directory
# (This assumes all necessary files are present in the submission directory

# Run the reproduction script
echo "Running SMM implementation..."
cd /home/submission
python3 smm_reproduction.py

echo "Reproduction completed successfully!"