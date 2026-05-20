#!/usr/bin/env bash
# Reproduce the BaM paper experiments
# This script installs the dependencies and runs the main experiment.
# The results are written to `results.csv` in the current directory.

set -euo pipefail
IFS=$'\n\t'

# Ensure we have a clean python environment
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet "jax==0.4.27" "jaxlib==0.4.27" optax tqdm numpy==1.26.4

# Run the experiment
python3 main.py --seed 42 --dim 16 --iterations 200 \
                --batch_size 256 --lambda_reg 1.0 \
                --advi_iter 200 --gsm_iter 200 \
                --output results.csv

echo "Reproduction finished. Results written to results.csv"