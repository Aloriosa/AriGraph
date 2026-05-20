#!/usr/bin/env bash
set -e

# Install dependencies
pip install --quiet -r requirements.txt

# Run the reproduction script
python -u src/main.py