#!/usr/bin/env bash
set -euo pipefail

# Update package lists and install python3
apt-get update && apt-get install -y python3 python3-pip

# Install python dependencies
pip install -r generated_reproduction.txt

# Run the evaluation script
python3 evaluate.py