#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing dependencies ==="
python -m pip install --quiet --upgrade pip
python -m pip install --quiet torch==2.2.0 numpy==1.26.4 pyro-ppl==1.8.0 tqdm

echo "=== Running the main experiment ==="
python src/main.py

echo "=== Done ==="