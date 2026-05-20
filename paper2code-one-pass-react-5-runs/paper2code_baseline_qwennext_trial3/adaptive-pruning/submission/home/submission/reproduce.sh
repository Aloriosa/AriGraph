#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip git wget

# Install required Python packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets accelerate peft scikit-learn numpy matplotlib tqdm

# Create working directory
mkdir -p /home/submission/code
cd /home/submission/code

# Download and extract the APT implementation
wget https://github.com/bowen98/apt/archive/refs/heads/main.zip -O apt.zip
unzip apt.zip
mv apt-main apt
cd apt

# Download required datasets and models
python3 download_data.py

# Run the main reproduction script
python3 train_apt.py --model_type roberta --task sst2 --sparsity 0.6 --epochs 5 --output_dir /home/submission/results

# Run additional experiments for comprehensive results
python3 train_apt.py --model_type t5 --task mnli --sparsity 0.6 --epochs 5 --output_dir /home/submission/results

# Run LLaMA experiment (with smaller model for resource constraints)
python3 train_apt.py --model_type llama --task alpaca --sparsity 0.3 --epochs 3 --output_dir /home/submission/results

# Generate evaluation results
python3 evaluate_results.py --input_dir /home/submission/results --output /home/submission/results/summary.csv

# Create final output files for grading
mkdir -p /home/submission/output
cp /home/submission/results/summary.csv /home/submission/output/
cp /home/submission/results/*.png /home/submission/output/

echo "APT reproduction completed. Results saved to /home/submission/output/"