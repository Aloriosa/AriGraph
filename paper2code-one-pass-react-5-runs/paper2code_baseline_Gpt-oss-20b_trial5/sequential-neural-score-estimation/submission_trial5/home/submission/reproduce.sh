#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
#  Install dependencies
# ------------------------------------------------------------------
echo "Installing Python dependencies..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

# ------------------------------------------------------------------
#  Run training and sampling
# ------------------------------------------------------------------
echo "Running NPSE toy experiment..."
python3 src/main.py

# ------------------------------------------------------------------
#  Done
# ------------------------------------------------------------------
echo "Reproduction finished. Posterior samples written to posterior_samples.csv"