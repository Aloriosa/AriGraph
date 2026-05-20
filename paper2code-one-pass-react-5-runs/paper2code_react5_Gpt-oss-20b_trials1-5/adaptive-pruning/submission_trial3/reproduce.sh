#!/usr/bin/env bash
set -euo pipefail

echo "Installing dependencies..."
pip install -q -U pip
pip install -q -U -r requirements.txt

echo "Running training..."
python src/training.py

echo "Reproduction finished."