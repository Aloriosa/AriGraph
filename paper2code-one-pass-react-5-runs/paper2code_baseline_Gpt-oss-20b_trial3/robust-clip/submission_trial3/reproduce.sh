#!/usr/bin/env bash
# ------------------------------------------------------------------
# Reproduction script for Robust CLIP (FARE)
# ------------------------------------------------------------------
set -euo pipefail

# Create a virtual environment (optional but keeps things clean)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Fine‑tune CLIP with FARE
python src/train_fare.py

# Evaluate the fine‑tuned model
python src/eval_fare.py

echo "Reproduction finished. Results are in results.txt"