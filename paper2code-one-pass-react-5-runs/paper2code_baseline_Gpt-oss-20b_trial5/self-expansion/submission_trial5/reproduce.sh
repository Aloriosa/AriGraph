#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# 1. Install dependencies
# ------------------------------------------------------------------
echo "Installing dependencies..."
python3 -m pip install --quiet -r requirements.txt

# ------------------------------------------------------------------
# 2. Run training script
# ------------------------------------------------------------------
echo "Running training..."
python3 -m src.train

# ------------------------------------------------------------------
# 3. Exit cleanly
# ------------------------------------------------------------------
echo "Reproduction finished. Results written to results.txt"