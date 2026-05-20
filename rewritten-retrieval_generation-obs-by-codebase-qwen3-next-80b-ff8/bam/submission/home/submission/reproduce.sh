#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv /home/submission/venv
source /home/submission/venv/bin/activate

# Install required packages
pip3 install torch numpy scipy matplotlib jax jaxlib optax

# Run the reproduction script
python3 reproduce.py

# Report completion
echo "Reproduction completed. Results saved in output/ directory."