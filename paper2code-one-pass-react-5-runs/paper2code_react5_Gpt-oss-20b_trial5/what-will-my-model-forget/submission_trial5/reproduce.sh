#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the training and evaluation pipeline
python -m src.train