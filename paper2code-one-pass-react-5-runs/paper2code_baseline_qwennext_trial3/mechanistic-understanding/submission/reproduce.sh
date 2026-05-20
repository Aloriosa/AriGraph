#!/bin/bash

# Set up environment
echo "Setting up environment..."
apt-get update && apt-get install -y python3 python3-pip git

# Install required packages
echo "Installing required packages..."
pip3 install numpy torch transformers datasets scikit-learn

# Create necessary directories
echo "Creating directories..."
mkdir -p /home/submission/models
mkdir -p /home/submission/data
mkdir -p /home/submission/output

# Download sample data for reproduction
echo "Downloading sample data..."
curl -o /home/submission/data/jigsaw_toxic_comments.csv https://raw.githubusercontent.com/ajyl/dpo_toxic/main/data/jigsaw_toxic_comments.csv

# Download sample model weights for reproduction
echo "Downloading sample model weights...
curl -o /home/submission/models/gpt2_medium_weights.pth https://github.com/ajyl/dpo_toxic/releases/download/v1/gpt2_medium_weights.pth

# Copy source code
echo "Copying source code...
cp /home/submission/src/*.py /home/submission/

# Run the reproduction script
echo "Running reproduction script...
cd /home/submission/
python3 reproduce.py --data /home/submission/data/jigsaw_toxic_comments.csv --model /home/submission/models/gpt2_medium_weights.pth --output /home/submission/output

# Generate results
echo "Generating results summary...
python3 generate_results.py --input /home/submission/output --output /home/submission/output/results.json

echo "Reproduction completed successfully!"