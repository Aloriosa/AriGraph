#!/usr/bin/env bash
# reproducible training script for SAPG (simplified toy version)

set -euo pipefail

# 1️⃣  Install dependencies
echo "Installing dependencies ..."
python3 -m pip install --quiet -U pip
python3 -m pip install --quiet -r requirements.txt

# 2️⃣  Train a toy SAPG agent on a vectorized CartPole environment
echo "Running training ..."
python3 src/train.py --env CartPole-v1 --num-envs 8 --policies 2 --steps 20000

echo "Training finished.  Logs and models are in ./results/"