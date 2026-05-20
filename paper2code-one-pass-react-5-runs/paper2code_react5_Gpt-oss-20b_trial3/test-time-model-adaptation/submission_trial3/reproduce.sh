#!/usr/bin/env bash
set -euo pipefail

# Make reproducible
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export TORCH_HOME=$HOME/.cache/torch
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions

# 1. Install dependencies
pip install -q -r requirements.txt --no-cache-dir

# 2. Run the experiment
python experiments/run_experiment.py