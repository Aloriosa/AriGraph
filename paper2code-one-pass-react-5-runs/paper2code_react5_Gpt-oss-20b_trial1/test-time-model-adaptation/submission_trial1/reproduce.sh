#!/usr/bin/env bash
set -euo pipefail

# Install system utilities
apt-get update -y
apt-get install -y python3-pip wget unzip

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the main script
python src/main.py