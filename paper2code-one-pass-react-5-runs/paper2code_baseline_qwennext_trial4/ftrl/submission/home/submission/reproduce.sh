#!/bin/bash
set -e

# Install required packages
apt-get update
apt-get install -y python3 python3-pip

# Install required Python packages
pip3 install gym numpy torch torchvision matplotlib

# Create directory structure
mkdir -p /home/submission/src
mkdir -p /home/submission/results

# Copy source files
cp /home/submission/src/*.py /home/submission/src/

# Run reproduction script
cd /home/submission/src

# Run the reproduction script
python3 reproduce.py

# Copy results to output directory
cp /home/submission/src/*.csv /home/submission/results/
cp /home/submission/src/*.png /home/submission/results/

echo "Reproduction completed successfully. Results saved in /home/submission/results/"