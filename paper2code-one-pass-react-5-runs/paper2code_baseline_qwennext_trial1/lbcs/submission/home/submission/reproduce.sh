#!/bin/bash
# Setup environment
apt-get update && apt-get install -y python3 python3-pip

# Install required packages
pip3 install torch torchvision numpy

# Create directory for code
mkdir -p /home/submission/src

# Copy the Python script to the source directory
cp /home/submission/src/count.py /home/submission/src/

# Run the reproduction script
cd /home/submission/src
python3 count.py --word strawberry --output output.csv

# Final message
echo "Reproduction complete. Output saved to output.csv"