#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install --quiet -r requirements.txt

# Create output directory
mkdir -p outputs

# Train SAPG
python train_sapg.py > outputs/sapg_log.txt
mv outputs/results_sapg.txt outputs/results_sapg.txt

# Train vanilla PPO baseline
python train_ppo.py > outputs/ppo_log.txt
mv outputs/results_ppo.txt outputs/results_ppo.txt

echo "Reproduction finished. Results are in the 'outputs/' directory."