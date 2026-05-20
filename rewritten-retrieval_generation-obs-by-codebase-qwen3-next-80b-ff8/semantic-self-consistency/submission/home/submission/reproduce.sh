#!/bin/bash
set -e

# Install dependencies
apt-get update && apt-get install -y python3 python3-pip git

# Install required Python packages
pip3 install torch transformers datasets accelerate sentencepiece tiktoken numpy scikit-learn torchmetrics

# Clone the dataset repositories if needed (SVAMP is from original source)
# We'll use Hugging Face datasets for GSM8K, MATH, ASDIV, and download SVAMP separately
mkdir -p /tmp/datasets
cd /tmp/datasets

# Download SVAMP dataset (from original source)
if [ ! -f "SVAMP.json" ]; then
    wget https://raw.githubusercontent.com/patil-suraj/exploring-the-limitations-of-bert-in-solving-arithmetic-word-problems/main/SVAMP/SVAMP.json
fi

# Create output directory
mkdir -p /home/submission/results

# Run the reproduction script
cd /home/submission
python3 main.py --datasets gsm8k math svamp asdiv --num_samples 10 --num_rationales 5 --output_dir /home/submission/results

echo "Reproduction complete. Results saved to /home/submission/results/"