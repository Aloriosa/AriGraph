#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install -q -r requirements.txt

# Run the full experiment
python full_pipeline.py