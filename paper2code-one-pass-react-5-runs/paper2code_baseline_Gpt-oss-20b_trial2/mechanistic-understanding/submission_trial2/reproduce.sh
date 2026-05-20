#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 1. Install system dependencies (only Python packages are needed)
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Train the DPO model
python3 train_dpo.py

# 3. Evaluate the fine‑tuned model
python3 evaluate.py

echo "Training completed."
echo "Evaluation finished."
echo "Results written to results.json"