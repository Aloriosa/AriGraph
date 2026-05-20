#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip

# Install required packages
pip3 install torch transformers datasets tqdm numpy

# Create directory for results
mkdir -p results

# Download and preprocess data
echo "Downloading and preprocessing data from PG-19 dataset..."
python3 preprocess_data.py --dataset pg19 --output data/pg19_sample.txt --sample_size 100

# Run compression experiment
echo "Running compression experiment..."
python3 compress_experiment.py --model Llama-3.1-8B --input data/pg19_sample.txt --output results/compression_results.json --max_tokens 1568 --num_vectors 1 --epochs 10 --learning_rate 0.01

# Generate results summary
echo "Generating results summary..."
python3 generate_summary.py --input results/compression_results.json --output results/summary.txt

echo "Reproduction complete. Results saved in results/ directory."