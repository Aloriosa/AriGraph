#!/usr/bin/env bash
set -e

# Install dependencies
pip install -r requirements.txt

# Ensure results directory exists
mkdir -p results

# Run the reproduction pipeline
python -m src.main --dataset sample_dataset.jsonl --output results/run_results.csv

echo "Reproduction finished. Results are in results/run_results.csv"