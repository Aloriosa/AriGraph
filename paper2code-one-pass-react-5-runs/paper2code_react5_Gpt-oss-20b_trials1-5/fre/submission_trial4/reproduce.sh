#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Install dependencies
pip install --quiet -r requirements.txt

# Run training and evaluation
python train_fre_and_policy.py --env antmaze-large-diverse-v2

echo "Reproduction complete. Results written to output.txt"