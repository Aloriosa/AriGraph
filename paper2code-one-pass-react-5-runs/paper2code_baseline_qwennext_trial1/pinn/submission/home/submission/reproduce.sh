#!/bin/bash
# This script reproduces the results from the paper "Challenges in Training PINNs: A Loss Landscape Perspective"

set -e  # Exit on any error

echo "Starting reproduction of PINN paper results..."

# Install required packages
echo "Installing required packages..."
apt-get update
apt-get install -y python3 python3-pip python3-numpy python3-matplotlib python3-scipy python3-torch python3-torchvision python3-torchaudio python3-pytest python3-h5py

# Install additional required packages
pip3 install torch torchvision torchaudio --upgrade
pip3 install numpy matplotlib scipy scikit-learn tqdm

# Create necessary directories
echo "Creating directories..."
mkdir -p /home/submission/models
mkdir -p /home/submission/results
mkdir -p /home/submission/plots

# Copy all source files
echo "Copying source files...
cp /home/submission/pinn.py /home/submission/
cp /home/submission/optimizer.py /home/submission/
cp /submission/README.md /home/submission/
cp /submission/requirements.txt /home/submission/

# Install requirements
echo "Installing Python requirements...
pip3 install -r /home/submission/requirements.txt

# Run the main reproduction script
echo "Running PINN training and optimization experiments...
cd /home/submission/
python3 pinn.py --pde wave --epochs 41000 --learning_rate 0.001 --output_dir /home/submission/results --seed 42

# Generate plots
echo "Generating plots...
python3 pinn.py --plot --input_dir /home/submission/results --output_dir /home/submission/plots

# Create summary file
echo "Creating summary file...
echo "Reproduction Summary" > /home/submission/results/summary.txt
echo "===================" >> /home/submission/results/summary.txt
echo "Reproduction completed at: $(date)" >> /home/submission/results/summary.txt
echo "Results saved to: /home/submission/results/" >> /home/submission/results/summary.txt

# Verify output files exist
echo "Verifying output files exist...
ls -la /home/submission/results/
ls -la /home/submission/plots/

echo "Reproduction completed successfully!"