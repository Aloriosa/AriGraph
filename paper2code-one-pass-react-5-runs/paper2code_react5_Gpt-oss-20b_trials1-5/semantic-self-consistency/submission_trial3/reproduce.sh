#!/usr/bin/env bash
set -e

# Install Python
apt-get update && apt-get install -y python3 python3-pip

# Install dependencies
pip3 install -r requirements.txt

# Run the reproduction script
python3 main.py --data_dir data --output results/results.csv

echo "Reproduction finished. Results stored in results/results.csv"