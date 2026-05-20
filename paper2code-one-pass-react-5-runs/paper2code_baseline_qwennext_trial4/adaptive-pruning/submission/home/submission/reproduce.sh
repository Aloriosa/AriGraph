#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip git wget

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets accelerate peft scikit-learn numpy matplotlib tqdm

# Download and prepare data
mkdir -p /home/submission/data
cd /home/submission/data

# Download a small subset of SST2 for demonstration (full dataset would be too large)
wget -O sst2.zip https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
unzip sst2.zip
mv SST-2/train.tsv train.tsv
mv SST-2/dev.tsv dev.tsv
rm -rf SST-2 sst2.zip

# Create output directory
mkdir -p /home/submission/results

# Run the APT implementation
cd /home/submission
python3 apt_main.py --model_type roberta --task sst2 --sparsity 0.6 --epochs 3 --batch_size 8 --output_dir /home/submission/results

# Run evaluation
python3 apt_eval.py --model_path /home/submission/results/roberta_sst2_pruned --task sst2 --output_file /home/submission/results/evaluation_results.json

# Generate summary report
python3 generate_report.py --results_dir /home/submission/results --output /home/submission/results/final_report.txt

echo "APT reproduction completed. Results saved to /home/submission/results/"