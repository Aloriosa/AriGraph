#!/usr/bin/env bash
set -euo pipefail

echo "Updating package lists..."
apt-get update -qq

echo "Installing Python 3 and pip..."
apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision tqdm numpy

# Create output directory
mkdir -p output

echo "Running LBCS experiments..."
python src/main.py --config config.json --output_dir output

echo "Reproduction finished. Results are saved in the 'output/' directory."