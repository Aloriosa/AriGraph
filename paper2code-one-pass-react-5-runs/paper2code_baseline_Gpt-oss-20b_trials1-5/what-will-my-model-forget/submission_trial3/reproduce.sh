#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the main pipeline
python src/main.py