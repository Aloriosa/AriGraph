#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Training pre‑trained policy ==="
python scripts/train_pretrained.py

echo "=== Training mask network ==="
python scripts/train_mask.py

echo "=== Refining policy with RICE ==="
python scripts/refine_agent.py

echo "=== Evaluating policies ==="
python scripts/evaluate.py

echo "=== Reproduction finished. Results are in logs/results.txt ==="