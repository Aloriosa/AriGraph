#!/usr/bin/env bash
set -e

# Install dependencies
pip install --upgrade pip
pip install -r generated_reproduction.txt

# Run the experiment
python src/main.py