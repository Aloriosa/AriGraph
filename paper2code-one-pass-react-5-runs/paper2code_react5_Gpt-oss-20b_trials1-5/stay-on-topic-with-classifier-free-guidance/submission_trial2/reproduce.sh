#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------
# 1. Install requirements
# -------------------------------------------------------------
pip install -q -U pip
pip install -q -r requirements.txt

# -------------------------------------------------------------
# 2. Run the demo generation script
# -------------------------------------------------------------
python generate.py