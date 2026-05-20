#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing dependencies ==="
# Use the system python (3.10+) which is available in the Docker image
pip install --quiet -r requirements.txt

echo "=== Training adapter ==="
python src/train.py

echo "=== Evaluating adapter ==="
python src/evaluate.py

echo "=== Reproduction finished ==="