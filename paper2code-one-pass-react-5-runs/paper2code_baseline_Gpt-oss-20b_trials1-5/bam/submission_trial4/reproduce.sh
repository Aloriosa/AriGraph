#!/usr/bin/env bash
set -e

# Install packages
python -m pip install -q -U pip
pip install -r requirements.txt

# Run the experiment
python src/main.py > output.log

# Show a short summary
echo "=== Summary ==="
tail -n 20 output.log