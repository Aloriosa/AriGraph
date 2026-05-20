#!/usr/bin/env bash
set -euo pipefail

# Ensure reproducibility
export PYTHONUNBUFFERED=1

# Install dependencies
if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found, aborting."
    exit 1
fi
pip install --upgrade pip
pip install -r requirements.txt

# Run the training script with default arguments
python train.py \
  --env CartPole-v1 \
  --timesteps 200000 \
  --eval-interval 20000

echo "Reproduction finished. Results are stored in output.csv."