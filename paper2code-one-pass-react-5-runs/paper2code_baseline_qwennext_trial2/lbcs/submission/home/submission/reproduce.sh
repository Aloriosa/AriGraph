#!/bin/bash
# Set up the environment
apt-get update && apt-get install -y python3 python3-pip

# Install required packages
pip3 install torch torchvision numpy

# Create the directory for the output
mkdir -p /home/submission/output

# Run the reproduction script
python3 /home/submission/leco.py --dataset mnist --k 1000 --epsilon 0.2 --output /home/submission/output/output.csv

# Inform the user that the output has been saved
echo "Refined Coreset Selection results saved to /home/submission/output/output.csv"