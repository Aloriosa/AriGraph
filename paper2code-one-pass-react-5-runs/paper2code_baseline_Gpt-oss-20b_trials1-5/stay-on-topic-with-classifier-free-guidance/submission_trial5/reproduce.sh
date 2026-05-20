#!/usr/bin/env bash
# reproduce.sh – reproduce the CFG experiment
set -euo pipefail

# 1. Install dependencies
echo "Installing python packages..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet torch==2.2.0 transformers datasets tqdm

# 2. Run the experiment
echo "Running CFG generation and perplexity experiment..."
python3 cfg_generate.py

echo "Reproduction finished. Check outputs.txt and perplexities.txt for results."