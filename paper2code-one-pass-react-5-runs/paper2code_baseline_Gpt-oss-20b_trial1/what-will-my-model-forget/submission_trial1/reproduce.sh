#!/usr/bin/env bash
set -euo pipefail

# 1. Install dependencies
echo "Installing dependencies..."
pip install --quiet -r requirements.txt

# 2. Run the training pipeline
echo "Fine‑tuning base model..."
python src/train.py

echo "Performing refinement and collecting forgetting data..."
python src/refine.py

echo "Training forecasting models..."
python src/forecast_train.py

echo "Evaluating forecasting models..."
python src/forecast_eval.py

# 3. Final message
echo "Reproduction finished."
echo "Results saved to outputs/metrics.json"