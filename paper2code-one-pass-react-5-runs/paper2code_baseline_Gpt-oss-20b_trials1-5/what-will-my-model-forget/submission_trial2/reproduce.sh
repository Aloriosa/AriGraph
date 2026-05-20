#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the main script
python src/main.py > log.txt 2>&1

# Summarise results
python - <<'PY'
import json, pathlib, sys
metrics = json.load(open('metrics.json'))
print("\n=== METRICS ===")
for k,v in metrics.items():
    print(f"{k:20s}: {v}")
PY