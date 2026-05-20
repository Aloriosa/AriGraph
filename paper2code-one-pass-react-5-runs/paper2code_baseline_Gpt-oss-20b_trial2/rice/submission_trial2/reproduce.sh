#!/usr/bin/env bash
set -e

# Update package list and install Python3
apt-get update -y
apt-get install -y python3 python3-pip

# Install Python dependencies
pip3 install -r requirements.txt

# Train the base agent
python3 train_base.py

# Collect critical states
python3 collect_critical.py

# Refine with RICE
python3 train_refine.py

echo "All done! Results written to results.json"