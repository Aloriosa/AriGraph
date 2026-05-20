#!/usr/bin/env bash
set -euo pipefail

# Install system packages (only python3 is required on the runner)
apt-get update && apt-get install -y python3 python3-pip

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Run the evaluation
python3 main.py --output results.txt

echo "Reproduction finished. Results written to results.txt"