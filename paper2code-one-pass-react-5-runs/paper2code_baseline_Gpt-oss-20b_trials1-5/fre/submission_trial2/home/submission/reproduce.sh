#!/usr/bin/env bash
set -euo pipefail

# 1. Install system packages
apt-get update && apt-get install -y python3-pip

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the full experiment
python run_experiment.py \
    --env CartPole-v1 \
    --offline_dataset offline_dataset.pt \
    --encoder_epochs 5 \
    --policy_epochs 5 \
    --device cuda \
    --seed 42

echo "Reproduction finished. Results written to results.json"