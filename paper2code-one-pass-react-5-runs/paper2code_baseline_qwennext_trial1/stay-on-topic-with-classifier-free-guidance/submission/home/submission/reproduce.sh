#!/bin/bash

# Set up environment
set -e

echo "Setting up environment for reproduction of 'Stay on topic with Classifier-Free Guidance'..."

# Install system dependencies
apt-get update && apt-get install -y python3 python3-pip python3-venv git curl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets evaluate scikit-learn numpy matplotlib seaborn pandas tqdm

echo "Dependencies installed successfully!"

# Download and prepare data
echo "Preparing datasets..."
mkdir -p data

# Download LAMBADA dataset
curl -o data/lambada_test.jsonl https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/data/lambada_test.jsonl
curl -o data/lambada_dev.jsonl https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/data/lambada_dev.jsonl

# Download HumanEval dataset
curl -o data/humaneval.json https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.json

# Download GPT-2 model weights
mkdir -p models
curl -o models/gpt2-medium.tar.gz https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin
curl -o models/gpt2-large.tar.gz https://huggingface.co/gpt2-large/resolve/main/pytorch_model.bin
curl -o models/llama-7b.tar.gz https://huggingface.co/huggyllama/llama-7b/resolve/main/pytorch_model.bin

# Extract models
mkdir -p models/gpt2-medium
tar -xzf models/gpt2-medium.tar.gz -C models/gpt2-medium
mkdir -p models/gpt2-large
tar -xzf models/gpt2-large.tar.gz -C models/gpt2-large
mkdir -p models/llama-7b
tar -xzf models/llama-7b.tar.gz -C models/llama-7b

echo "Data and models prepared successfully!"

# Run reproduction script
echo "Running reproduction script..."
python reproduce.py

echo "Reproduction completed successfully!"

# Output results
echo "Results:"
ls -la results/

# Print final result summary
echo "Final result summary:"
cat results/summary.txt