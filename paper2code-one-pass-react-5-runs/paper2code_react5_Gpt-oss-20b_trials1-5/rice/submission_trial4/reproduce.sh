#!/usr/bin/env bash
set -e

# Install Python dependencies
pip install -r requirements.txt

# Run the training and evaluation script
python -m src.train

echo "Reproduction finished. Results are stored in output.txt"