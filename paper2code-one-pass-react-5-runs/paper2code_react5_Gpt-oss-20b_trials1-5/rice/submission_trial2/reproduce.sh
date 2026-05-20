#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install -q -U pip
pip install -r requirements.txt

# 1. Train a baseline PPO agent
python src/train_pretrained.py

# 2. Train the mask network
python src/train_mask.py

# 3. Train the refined agent (RICE)
python src/train_refine.py