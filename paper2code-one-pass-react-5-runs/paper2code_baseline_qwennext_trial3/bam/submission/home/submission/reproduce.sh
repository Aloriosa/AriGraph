#!/bin/bash
# Reproduction script for "Batch and match: black-box variational inference with a score-based divergence"

set -e  # Exit on any error

echo "Setting up environment for reproduction..."

# Install system dependencies
apt-get update && apt-get install -y python3 python3-pip python3-venv git

# Create virtual environment
python3 -m venv /tmp/reproduce_venv
source /tmp/reproduce_venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install numpy scipy matplotlib jax jaxlib

# Create output directory
mkdir -p /home/submission/output

echo "Running Batch and Match algorithm reproduction..."

# Run the main reproduction script
cd /home/submission
python3 batch_and_match.py --iterations 50 --batch_size 10 --dimension 10 --output /home/submission/output/results.json

echo "Reproduction completed successfully!"

# Generate a summary report
python3 generate_report.py /home/submission/output/results.json /home/submission/output/report.md

echo "Summary report generated at /home/submission/output/report.md"