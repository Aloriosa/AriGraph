#!/usr/bin/env bash
# ------------------------------------------------------------
# Reproduction script for “Semantic Self‑Consistency: Enhancing
# Language Model Reasoning via Semantic Weighting”.
# ------------------------------------------------------------
set -euo pipefail

# 1. Install system packages
apt-get update -y
apt-get install -y python3 python3-pip git

# 2. Upgrade pip
python3 -m pip install --upgrade pip

# 3. Install the Python dependencies
python3 -m pip install -r requirements.txt

# 4. Run the experiment
python3 run_experiment.py