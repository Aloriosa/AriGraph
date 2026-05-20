#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip git libgl1-mesa-glx libglib2.0-0

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy matplotlib tqdm scikit-learn pandas pyyaml

# Create directory structure
mkdir -p /home/submission/data
mkdir -p /home/submission/models
mkdir -p /home/submission/results

# Run LBCS coreset selection and evaluation on all datasets
python3 main.py --dataset fmnist --coreset_size 1000 --performance_threshold 0.95 --output results/fmnist --seed 42
python3 main.py --dataset svhn --coreset_size 1000 --performance_threshold 0.95 --output results/svhn --seed 42
python3 main.py --dataset cifar10 --coreset_size 4000 --performance_threshold 0.95 --output results/cifar10 --seed 42

# Generate summary report
python3 summarize_results.py --input_dir results --output results/summary.csv

# Report results
echo "Reproduction complete!"
echo "Results saved to results/fmnist/, results/svhn/, results/cifar10/"
echo "Summary saved to results/summary.csv"