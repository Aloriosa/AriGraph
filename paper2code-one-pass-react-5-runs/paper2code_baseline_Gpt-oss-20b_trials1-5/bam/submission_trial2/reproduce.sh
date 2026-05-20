#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies
python3 -m pip install --quiet -U pip
pip install --quiet -r requirements.txt

# Run the experiment
python3 main.py