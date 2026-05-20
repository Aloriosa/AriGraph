#!/usr/bin/env bash
set -euo pipefail

# Update package lists and install Python3
apt-get update && apt-get install -y python3 python3-pip

# Install Python dependencies
pip3 install -r requirements.txt

# Run the evaluation script
python3 evaluate.py > log.txt

echo "Reproduction finished. Results are in results.csv and log.txt"