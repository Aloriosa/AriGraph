#!/usr/bin/env bash
# -------------------------------------------------------------
# Reproduce script for the BaM implementation.
# -------------------------------------------------------------
set -euo pipefail

# Install dependencies
echo "Installing Python dependencies..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet numpy scipy

# Run the BaM demo
echo "Running BaM demo..."
python3 baM.py > results.txt

# Print results
echo "=== Reproduction Output ==="
cat results.txt
echo "=== End of Output ==="