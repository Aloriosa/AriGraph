#!/bin/bash

# Set up the environment for reproduction
echo "Setting up environment for Simformer reproduction..."

# Update package list and install required system dependencies
apt-get update
apt-get install -y python3 python3-pip python3-venv git

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install numpy==1.26.0 torch==2.1.0 scikit-learn==1.3.0 matplotlib==3.7.3

# Create directory structure
echo "Creating directory structure..."
mkdir -p /home/submission/output
mkdir -p /home/submission/models

# Download and extract the Simformer implementation
echo "Downloading Simformer implementation...
cd /home/submission
wget https://github.com/mackelab/simformer/archive/refs/heads/main.zip
unzip main.zip
mv simformer-main /home/submission/simformer

# Copy our implementation files
echo "Copying implementation files...
cp /home/submission/simformer/simformer.py /home/submission/simformer.py
cp /home/submission/simformer/reproduce.py /home/submission/reproduce.py
cp /home/submission/simformer/config.py /home/submission/config.py

# Run the reproduction script
echo "Running reproduction script...
cd /home/submission
python reproduce.py

# Generate summary report
echo "Generating summary report...
python -c "
import os
import numpy as np
import pandas as pd

# Create summary report
with open('summary.txt', 'w') as f:
    f.write('Simformer Reproduction Summary\n')
    f.write('='*50 + '\n\n')
    
    # Check output files
    output_files = ['output/posterior_samples.csv', 'output/likelihood_samples.csv', 'output/conditional_samples.csv', 'output/interval_constraints.csv']
    
    for file in output_files:
        if os.path.exists(file):
            f.write(f'✓ {file}: Generated\n')
        else:
            f.write(f'✗ {file}: Missing\n'
    
    # Check model files
    model_files = ['models/simformer_model.pth']
    for file in model_files:
        if os.path.exists(file):
            f.write(f'✓ {file}: Generated\n')
        else:
            f.write(f'✗ {file}: Missing\n'
    
    f.write('\n')

# Print summary
echo "Reproduction complete!"
echo "Check summary.txt for details on generated files."

# Clean up
echo "Cleaning up..."
rm -rf /home/submission/simformer-main
rm -rf /home/submission/main.zip

echo "Reproduction completed successfully!"