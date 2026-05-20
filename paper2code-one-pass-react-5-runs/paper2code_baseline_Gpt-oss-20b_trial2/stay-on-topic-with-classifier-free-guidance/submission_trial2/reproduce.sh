#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# 1. Install Python dependencies
# ------------------------------------------------------------------
echo "Installing Python dependencies..."
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3-pip python3-venv

export VENV_DIR="/home/submission/venv"
python3 -m venv "$VENV_DIR"
# Activate virtual environment
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install torch==2.3.0 transformers==4.41.2
# --------------------------------------------------------------


# ------------------------------------------------------------------
# 2. Run the demo script
# ------------------------------------------------------------------
echo "Running baseline generation..."
python cfg_demo.py --model gpt2 --prompt-file prompts.txt \
    --output baseline.txt --gamma 1.0

echo "Running CFG generation..."
python cfg_demo.py --model gpt2 --prompt-file prompts.txt \
    --output cfg.txt --gamma 1.5

echo "Reproduction finished."