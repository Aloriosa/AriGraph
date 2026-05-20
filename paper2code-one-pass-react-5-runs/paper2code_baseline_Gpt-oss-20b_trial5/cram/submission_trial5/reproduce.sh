#!/usr/bin/env bash
set -euo pipefail

# Install required packages
python3 -m pip install --quiet torch==2.1.0 transformers==4.40.0 tqdm==4.66.1

# Run the compression experiment
python3 src/compress.py \
    --model gpt2 \
    --text "The quick brown fox jumps over the lazy dog." \
    --max_steps 200 \
    --output results.json