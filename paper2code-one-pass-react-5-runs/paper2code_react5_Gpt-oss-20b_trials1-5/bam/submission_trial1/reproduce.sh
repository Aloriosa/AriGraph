#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
echo "Installing Python dependencies..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install -r requirements.txt

# Run the experiment
echo "Running BaM experiment..."
python3 examples/run_gaussian.py > results/output.txt

echo "Reproduction completed. Results can be found in the 'results/' directory:"
echo "  - output.txt        : console output"
echo "  - baum_convergence.png : convergence plot"