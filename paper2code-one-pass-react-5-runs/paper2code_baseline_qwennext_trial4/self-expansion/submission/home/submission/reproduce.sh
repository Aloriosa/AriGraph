#!/bin/bash
# This script reproduces the results from the SEMA paper
# It trains the SEMA model on CIFAR-100 dataset and reports accuracy

# Set up environment
apt-get update
apt-get install -y python3 python3-pip python3-venv git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install torch torchvision numpy matplotlib scikit-learn tqdm

# Download and extract the dataset if needed
mkdir -p data
cd data

# Download CIFAR-100 dataset
if [ ! -f "cifar-100-python.tar.gz" ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
fi

# Extract dataset
if [ ! -d "cifar-100-python" ]; then
    tar -xzf cifar-100-python.tar.gz
fi

cd ..

# Copy source code
mkdir -p src
cp -r src/* src/

# Run the reproduction script
cd /home/submission/src
python3 main.py --dataset cifar100 --epochs 10 --batch_size 32 --learning_rate 0.005 --output_dir /home/submission/results

# Print completion message
echo "Reproduction completed. Results saved in /home/submission/results"

# Check if results file was created
if [ -f "/home/submission/results/accuracy_results.txt" ]; then
    echo "Results file created successfully!"
    cat /home/submission/results/accuracy_results.txt
else
    echo "ERROR: Results file not created!"
    exit 1
fi