#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# 1. Install dependencies
# ------------------------------------------------------------
echo "Installing Python dependencies..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

# ------------------------------------------------------------
# 2. Train the adapter
# ------------------------------------------------------------
echo "Starting training..."
python3 src/train.py

# ------------------------------------------------------------
# 3. Run inference on the test set
# ------------------------------------------------------------
echo "Running inference..."
python3 src/inference.py

# ------------------------------------------------------------
# 4. Evaluate the predictions
# ------------------------------------------------------------
echo "Evaluating predictions..."
python3 src/evaluate.py

echo "Reproduction finished."