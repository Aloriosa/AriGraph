#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the training script
# The script will create ./logs/ and write a final evaluation file.
python -m src.main --env Hopper-v3 --timesteps 200000 --seed 1234 --device cpu --mix_prob 0.25 --rnd_coef 0.01 --alpha 0.01

# Print summary
echo "=== RICE reproduction finished ==="
cat ./logs/evaluation.txt