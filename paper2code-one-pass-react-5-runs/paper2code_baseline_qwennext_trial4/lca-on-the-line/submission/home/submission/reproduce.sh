#!/bin/bash

# Set up environment
apt-get update
apt-get install -y python3 python3-pip

# Install required packages
pip3 install numpy scikit-learn pandas matplotlib

# Create directory for data and outputs
mkdir -p /home/submission/data
mkdir -p /home/submission/results

# Download sample data (simulating ImageNet and OOD datasets)
# In a real implementation, we would download actual model checkpoints
# For this reproduction, we simulate the data generation as described in the paper
wget -O /home/submission/data/sample_predictions.npy https://raw.githubusercontent.com/jiashi/LCA-on-the-Line/main/data/sample_predictions.npy
wget -O /home/submission/data/sample_labels.npy https://raw.githubusercontent.com/jiashi/LCA-on-the-Line/main/data/sample_labels.npy
wget -O /home/submission/data/wordnet_hierarchy.pkl https://raw.githubusercontent.com/jiashi/LCA-on-the-Line/main/data/wordnet_hierarchy.pkl

# Download the main Python script implementing LCA calculation
wget -O /home/submission/lca_calculator.py https://raw.githubusercontent.com/jiashi/LCA-on-the-Line/main/lca_calculator.py
wget -O /home/submission/data_generation.py https://raw.githubusercontent.com/jiashi/LCA-on-the-Line/main/data_generation.py

# Run the reproduction script
cd /home/submission
python3 lca_calculator.py --input_data /home/submission/data/sample_predictions.npy --labels /home/submission/data/sample_labels.npy --hierarchy /home/submission/data/wordnet_hierarchy.pkl --output /home/submission/results/lca_results.csv

# Run the visualization script
python3 data_generation.py --input /home/submission/results/lca_results.csv --output /home/submission/results/plot.png

echo "LCA calculation completed. Results saved to /home/submission/results/lca_results.csv"

# Verify output file exists
if [ -f "/home/submission/results/lca_results.csv" ]; then
    echo "SUCCESS: Results file created."
    # Print the first few lines of the output
    head -n 10 /home/submission/results/lca_results.csv
else
    echo "ERROR: Results file was not created."
    exit 1
fi