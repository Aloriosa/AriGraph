#!/usr/bin/env bash
set -euo pipefail

# Install python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the full experiment (all three datasets)
python src/main.py --dataset all