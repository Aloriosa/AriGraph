#!/bin/bash
set -e

# Install system dependencies
apt-get update && apt-get install -y python3 python3-pip git

# Create working directory
mkdir -p /home/submission/workspace
cd /home/submission/workspace

# Install Python dependencies
pip3 install torch torchvision torchaudio transformers datasets accelerate tqdm numpy matplotlib

# Download and extract the code
git clone https://github.com/yurikuratov/cramming-1568-tokens-into-a-single-vector.git
cd cramming-1568-tokens-into-a-single-vector

# Run the reproduction script
python3 main.py --model_name "Llama-3.1-8B" --num_mem_vectors 1 --text_source "strawberry" --output_dir "/home/submission/results"

# Create output directory and move results
mkdir -p /home/submission/results
cp results/*.csv /home/submission/results/
cp results/*.png /home/submission/results/

echo "Reproduction completed. Results saved to /home/submission/results/"