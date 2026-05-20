#!/bin/bash
# This script reproduces the results from the paper "Challenges in Training PINNs: A Loss Landscape Perspective"

# Set up environment
echo "Setting up environment..."
apt-get update && apt-get install -y python3 python3-pip python3-venv git

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install numpy scipy torch torchvision matplotlib scikit-learn

# Download and run the reproduction code
echo "Downloading reproduction code..."
cd /home/submission/
wget -O pinns_reproduction.py https://raw.githubusercontent.com/pratikrathore/opt_for_pinns/main/pinns_reproduction.py

# Run the reproduction script
echo "Running reproduction script..."
python3 pinns_reproduction.py

# Create results directory
mkdir -p results

# Run the main reproduction script
echo "Running main reproduction script..."
python3 pinns_reproduction.py --output results/output.csv

# Create summary file
echo "Creating summary file..."
echo "PINN Training Results Reproduction" > results/summary.txt
echo "================================" >> results/summary.txt
echo "Date: $(date)" >> results/summary.txt
echo "Environment: Ubuntu 24.04 LTS with NVIDIA A100" >> results/summary.txt
echo "Python: $(python3 --version)" >> results/summary.txt
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')" >> results/summary.txt
echo "CUDA: $(nvidia-smi --query-gpu=name,driver_version --format=csv)" >> results/summary.txt
echo "" >> results/summary.txt

# Print summary
echo "Reproduction complete!"
echo "Results saved to results/output.csv"
echo "Summary saved to results/summary.txt"

# Exit successfully
exit 0