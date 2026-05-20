#!/bin/bash
# This script reproduces the results from the paper "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem"

# Set up the environment
echo "Setting up the environment..."
apt-get update && apt-get install -y python3 python3-pip

# Install required packages
echo "Installing required packages..."
pip3 install numpy torch torchvision matplotlib scikit-learn

# Create the necessary directories
echo "Creating directories..."
mkdir -p /home/submission/models
mkdir -p /home/submission/results

# Run the reproduction code
echo "Running the reproduction code..."
cd /home/submission
python3 reproduce.py

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Reproduction completed successfully!"
else
    echo "Reproduction failed!"
    exit 1
fi

# Generate a summary of the results
echo "Generating summary of results..."
python3 summarize_results.py

echo "Reproduction complete!"