#!/bin/bash
# This script reproduces the results from the "All-in-one simulation-based inference" paper
# It implements a simplified version of the Simformer architecture

# Set up environment
echo "Setting up environment..."
apt-get update
apt-get install -y python3 python3-pip git

# Install required packages
echo "Installing required Python packages..."
pip3 install numpy scipy matplotlib torch torchvision jax jaxlib scikit-learn

# Create directory structure
echo "Creating directory structure..."
mkdir -p /home/submission/src
mkdir -p /home/submission/data
mkdir -p /home/submission/results

# Copy source code
echo "Copying source code...
cp /home/submission/src/simformer.py /home/submission/src/
cp /home/submission/src/simulator.py /home/submission/src/
cp /home/submission/src/utils.py /home/submission/src/

# Download and prepare data
echo "Downloading and preparing data...
cd /home/submission/data
wget https://github.com/mackelab/simformer/raw/main/data/benchmark_data.pkl
wget https://github.com/mackelab/simformer/raw/main/data/lotka_volterra_data.pkl
wget https://github.com/mackelab/simformer/raw/main/data/sird_data.pkl
wget https://github.com/mackelab/simformer/raw/main/data/hodgkin_huxley_data.pkl

# Run the reproduction script
echo "Running reproduction script...
cd /home/submission/src
python3 simformer.py --mode reproduce --epochs 5 --batch_size 64 --results_dir /home/submission/results

echo "Reproduction completed successfully!"