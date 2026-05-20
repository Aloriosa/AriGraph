#!/usr/bin/env bash
set -e

# Install dependencies
pip install -r requirements.txt

# Run training
python train_apt.py