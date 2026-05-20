#!/usr/bin/env bash
# ------------------------------------------------------------------
# reproduce.sh
# --------------
# This script sets up a minimal environment and reproduces a toy
# implementation of the stochastic interpolants with data‑dependent
# couplings described in "Stochastic Interpolants with Data‑Dependent Couplings".
#
# The script:
#   1. Creates a virtual environment.
#   2. Installs the required Python packages.
#   3. Trains the velocity model on CIFAR‑10 (few epochs).
#   4. Generates 20 samples (in‑painted images) and saves them in ./samples/
# ------------------------------------------------------------------

set -euo pipefail

# 1. Create a virtual environment (optional, keeps things clean)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -q -U pip
pip install -r requirements.txt

# 3. Train the model
python train.py --epochs 10 --batch-size 128 --lr 2e-4 --save-model model.pt

# 4. Generate samples
python sample.py --model model.pt --num-samples 20 --output samples

echo "Reproduction finished. Generated images are in the ./samples/ directory."