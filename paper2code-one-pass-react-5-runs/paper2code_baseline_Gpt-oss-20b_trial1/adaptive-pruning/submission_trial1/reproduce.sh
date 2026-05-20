#!/usr/bin/env bash
set -euo pipefail

# Create a clean environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download data (SST-2) – HuggingFace datasets will handle this
python -c "from datasets import load_dataset; load_dataset('glue', 'sst2', split='train[:1%]')"

# Run training
python src/train.py

# Capture final accuracy
python - <<'PY'
import json, os
with open('output/accuracy.txt') as f:
    print(f"\n=== Final Accuracy ===\n{f.read()}")
PY