#!/usr/bin/env bash
set -e

# 1. Install Python 3 and pip
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip

# 2. Install required Python packages
pip3 install --upgrade pip
pip3 install -r requirements.txt

# 3. Train FRE and evaluate
python3 -m src.trainer
python3 -m src.eval

echo "=== All done. Results are in experiments/results.txt ==="
cat experiments/results.txt