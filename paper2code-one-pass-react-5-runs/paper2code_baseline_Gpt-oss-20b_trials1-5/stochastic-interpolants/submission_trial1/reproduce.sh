#!/usr/bin/env bash
# Reproduce the training and sampling of the minimal stochastic interpolant model.

set -e

# Install python packages
pip install -r requirements.txt

# Run training and sampling
python -u main.py