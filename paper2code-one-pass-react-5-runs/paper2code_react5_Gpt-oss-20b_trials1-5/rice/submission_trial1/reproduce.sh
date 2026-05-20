#!/usr/bin/env bash
set -euo pipefail

# Update package lists and install system dependencies
apt-get update && apt-get install -y python3-pip git

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the main script
python3 main.py

echo "Reproduction finished. Final reward written to final_reward.txt"