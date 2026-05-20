#!/bin/bash

# Install required packages
apt-get update && apt-get install -y python3 python3-pip

# Install Python dependencies
pip3 install numpy scikit-learn torch torchvision

# Create directory for results
mkdir -p results

# Run the LBCS implementation
python3 lbc_s.py --dataset mnist --k 1000 --epsilon 0.2 --output results/output.csv

# Print completion message
echo "LBCS implementation completed. Results saved to results/output.csv"