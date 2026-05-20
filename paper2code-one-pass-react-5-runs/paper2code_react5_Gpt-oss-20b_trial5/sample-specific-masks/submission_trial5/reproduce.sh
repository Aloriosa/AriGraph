#!/usr/bin/env bash
set -e

# Install dependencies
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet torch torchvision timm

# Run training
python3 src/train_smm.py

# Print final accuracy report
echo "Reproduction finished. See results.txt for test accuracies."