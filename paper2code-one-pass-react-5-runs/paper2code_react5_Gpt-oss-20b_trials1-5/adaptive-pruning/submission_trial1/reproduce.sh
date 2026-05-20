#!/usr/bin/env bash
set -e

# Install Python dependencies
python -m pip install -q -U pip
pip install -r requirements.txt

# Run the APT training script
python train_apt.py