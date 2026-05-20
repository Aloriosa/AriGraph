#!/bin/bash
# Reproduction script for "Batch and match: black-box variational inference with a score-based divergence"

set -e  # Exit on any error

echo "Starting reproduction of 'Batch and match: black-box variational inference with a score-based divergence'"

# Install dependencies
echo "Installing required packages..."
apt-get update
apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install numpy scipy matplotlib jax jaxlib

# Copy all source files from submission directory
echo "Copying source files to submission directory..."
mkdir -p /home/submission/src
cp -r /home/submission/src/* /home/submission/src/

# Run experiments
echo "Running BaM experiments..."
cd /home/submission/src
python3 main.py --output_dir /home/submission/results

echo "Experiments completed successfully!"

# Generate summary report
echo "Generating summary report..."
python3 generate_report.py --results_dir /home/submission/results --output /home/submission/results/report.md

echo "Reproduction completed successfully!"