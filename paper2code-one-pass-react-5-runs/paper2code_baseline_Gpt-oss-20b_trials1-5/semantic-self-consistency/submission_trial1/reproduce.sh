#!/usr/bin/env bash
set -euo pipefail

echo "Updating package lists..."
apt-get update -y

echo "Installing Python 3 and pip..."
apt-get install -y python3 python3-pip

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Running evaluation..."
python3 src/evaluate.py

echo "Reproduction finished."