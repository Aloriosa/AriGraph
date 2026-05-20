#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv /home/submission/venv
source /home/submission/venv/bin/activate

# Install required packages
pip install numpy scikit-learn torch transformers datasets pandas matplotlib

# Download and prepare datasets (simulated)
python3 prepare_datasets.py

# Run the main reproduction script
python3 semantic_self_consistency.py

# Generate plots and summary statistics
python3 generate_results.py

echo "Reproduction complete. Results generated in results/ directory."