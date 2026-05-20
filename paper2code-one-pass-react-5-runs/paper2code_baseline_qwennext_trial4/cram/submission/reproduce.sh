#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip

# Install required packages
pip3 install torch transformers datasets numpy

# Create directory for outputs
mkdir -p /home/submission/output

# Download and run the reproduction script
cd /home/submission
python3 compress_text.py --model_name "meta-llama/Meta-Llama-3-8B" --text_source "pg19" --num_mem_vectors 1 --output_dir "/home/submission/output"

echo "Reproduction script completed. Output saved to /home/submission/output/"