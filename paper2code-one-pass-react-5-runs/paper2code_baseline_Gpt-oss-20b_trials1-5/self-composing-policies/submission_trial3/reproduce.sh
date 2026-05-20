#!/usr/bin/env bash
set -euo pipefail

# 1. Install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Create results directory
mkdir -p results

# 3. Run training for each benchmark
python3 train.py --benchmark meta-world
python3 train.py --benchmark spaceinvaders
python3 train.py --benchmark freeway

echo "Reproduction finished. Results in ./results/"