#!/usr/bin/env bash
# ------------------------------------------------------------------
# Reproduce BaM experiment
# ------------------------------------------------------------------
set -euo pipefail

# Install Python dependencies
if ! command -v pip &>/dev/null; then
    echo "pip not found, installing via get-pip.py"
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
fi

pip install -q -r requirements.txt

# Run experiment
python3 experiments/gaussian_experiment.py