#!/usr/bin/env bash
set -euo pipefail

# Update packages and install system dependencies
apt-get update -qq
apt-get install -y -qq python3-pip

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Run the experiment in all three modes
python run_apt.py --mode apt
python run_apt.py --mode lora
python run_apt.py --mode prune

# Collect results into a summary table
python summarize_results.py