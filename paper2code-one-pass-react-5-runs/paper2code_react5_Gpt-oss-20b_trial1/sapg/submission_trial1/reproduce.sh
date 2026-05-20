#!/usr/bin/env bash
set -e

# Update package lists and install python3-venv
apt-get update -y
apt-get install -y python3-pip python3-venv

# Create a virtual environment
python3 -m venv sapg-env
source sapg-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Run training
python train_sapg.py > logs/training.log 2>&1

# Print final results
echo "=== Final Results ==="
cat logs/final_results.txt