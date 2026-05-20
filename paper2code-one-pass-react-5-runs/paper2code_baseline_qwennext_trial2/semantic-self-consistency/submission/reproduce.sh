#!/bin/bash

# Reproduce the results from the paper "Semantic Self-Consistency: Enhancing Language Model Reasoning via Semantic Weighting"

# Exit on any error
set -e

# Set up environment
echo "Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:/home/submission/src"
mkdir -p /home/submission/data
mkdir -p /home/submission/models

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y python3 python3-pip git

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r /home/submission/requirements.txt

# Download datasets (if not already downloaded)
echo "Downloading datasets..."
python3 /home/submission/src/dataset_loader.py --download

# Run the main reproduction script
echo "Running semantic self-consistency reproduction..."
python3 /home/submission/src/reproduce_main.py

# Print completion message
echo "Reproduction complete. Results saved to /home/submission/results.json"

# Output results summary
echo "=== RESULTS SUMMARY ==="
python3 /home/submission/src/print_results.py

# Exit successfully
exit 0