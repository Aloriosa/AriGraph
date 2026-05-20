#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install --quiet -r requirements.txt

# Run the full pipeline
python src/main.py