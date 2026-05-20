#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing dependencies ==="
pip install -q -U pip
pip install -q -r requirements.txt

echo "=== Training adapter ==="
python -m src.train_adapter

echo "=== Training and evaluation completed ==="
echo "Check models/eval.txt for the final accuracy."