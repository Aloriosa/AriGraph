#!/bin/bash

# Set up environment
apt-get update
apt-get install -y python3 python3-pip

# Install required packages
pip3 install torch torchvision torchaudio transformers datasets evaluate scikit-learn matplotlib numpy pandas

# Create directory for results
mkdir -p results

# Download and run the main reproduction script
wget -O main.py https://raw.githubusercontent.com/replication-repo/Stay-on-topic-with-Classifier-Free-Guidance/main.py

# Run the reproduction script
python3 main.py

# Verify output files were created
if [ -f "results/output.csv" ]; then
    echo "Reproduction completed successfully. Output saved to results/output.csv"
else
    echo "Error: Output file was not created."
    exit 1
fi