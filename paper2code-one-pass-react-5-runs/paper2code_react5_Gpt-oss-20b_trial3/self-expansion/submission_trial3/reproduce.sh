#!/usr/bin/env bash
set -e

# Install dependencies
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet torch torchvision transformers tqdm

# Run training and evaluation
python3 train_sema.py --tasks 10 --batch 32 --expansions 3

echo "Reproduction finished. Results written to results.txt"