#!/usr/bin/env bash
set -euo pipefail

# Install python dependencies
pip install --quiet -r requirements.txt

# Run the training & evaluation script
python src/fare_clip.py