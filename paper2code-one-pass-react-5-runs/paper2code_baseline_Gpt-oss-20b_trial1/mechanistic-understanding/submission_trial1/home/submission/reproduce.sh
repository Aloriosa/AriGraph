#!/usr/bin/env bash
set -e

# Install dependencies
pip install -r requirements.txt

# Make sure we are using the GPU if available
export CUDA_VISIBLE_DEVICES=0

# Run the full reproduction pipeline
python main.py