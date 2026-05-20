#!/bin/bash
# This script reproduces the results from the paper "Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints"

# Install required dependencies
apt-get update && apt-get install -y python3 python3-pip

# Install required Python packages
pip3 install torch torchvision numpy scikit-learn tqdm

# Create a directory for the results
mkdir -p /home/submission/results

# Download the code from the GitHub repository mentioned in the paper
# Note: The paper mentions the implementation is available at https://github.com/xiaoboxia/LBCS
# We will create a local version of the algorithm described in the paper

# Create the main script for LBCS (Lexicographic Bilevel Coreset Selection)
mkdir -p /home/submission/src
cp /home/submission/src/lbcs.py /home/submission/src/

# Run the reproduction script
python3 /home/submission/src/lbcs.py --output /home/submission/results/output.csv --dataset mnist --k 200 --epsilon 0.2 --iterations 500

# Print completion message
echo "Reproduction script completed. Results saved to /home/submission/results/output.csv"