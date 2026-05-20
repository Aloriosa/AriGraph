#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip git

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets accelerate peft scikit-learn numpy matplotlib tqdm

# Clone the repository (if needed) - we're already in the submission directory
cd /home/submission

# Run the APT reproduction script
python3 apt_reproduction.py --model_type roberta --task sst2 --sparsity 0.6 --epochs 5 --batch_size 16 --output_dir /home/submission/results

# Run additional experiments for completeness
python3 apt_reproduction.py --model_type t5 --task mnli --sparsity 0.6 --epochs 5 --batch_size 16 --output_dir /home/submission/results

# Generate performance summary
python3 generate_summary.py

echo "APT reproduction completed. Results saved in /home/submission/results"