#!/usr/bin/env bash
set -euo pipefail

# Update package lists and install Python3
sudo apt-get update -y && sudo apt-get install -y python3 python3-pip

# Install Python dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Create results directory
mkdir -p results

# Run training scripts
echo "===== Training Two‑State MDP ====="
python3 src/train_mdp.py --save results/mdp_results.csv

echo "===== Training AppleRetrieval ====="
python3 src/train_apple.py --save results/apple_results.csv

echo "===== Reproduction finished ====="
echo "Results written to ./results/"