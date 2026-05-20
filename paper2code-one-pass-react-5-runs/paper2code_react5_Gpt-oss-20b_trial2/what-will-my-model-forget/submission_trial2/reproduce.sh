#!/usr/bin/env bash
set -euo pipefail

# Install dependencies (only if not already installed)
pip install --quiet -r requirements.txt

# Run the main script
python main.py > metrics.txt 2>&1

echo "Reproduction finished. Results written to metrics.txt"