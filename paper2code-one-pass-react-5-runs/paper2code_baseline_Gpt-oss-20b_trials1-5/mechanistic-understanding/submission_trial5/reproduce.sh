#!/usr/bin/env bash
set -euo pipefail

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Training DPO model..."
python src/train_dpo.py

echo "Evaluating model..."
python src/evaluate.py

echo "Reproduction complete. Results are in results.json"