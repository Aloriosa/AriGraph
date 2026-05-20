#!/usr/bin/env bash
set -e

# Choose dataset: cifar100 | imagenet_r | imagenet_a | vtab
DATASET="cifar100"

# Create logs dir
mkdir -p logs

# Run training
python src/train.py --dataset "$DATASET" --output_dir "logs/${DATASET}"