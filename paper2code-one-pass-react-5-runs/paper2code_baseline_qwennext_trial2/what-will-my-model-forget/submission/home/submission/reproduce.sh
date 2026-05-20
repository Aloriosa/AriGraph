#!/bin/bash

# Set up the environment
echo "Setting up environment..."
mkdir -p output models

# Install required packages
echo "Installing required packages..."
apt-get update && apt-get install -y python3 python3-pip

# Install Python dependencies
pip3 install torch torchvision torchaudio numpy scikit-learn pandas tqdm

# Download and extract data if needed (simulated data)
echo "Generating simulated dataset..."
python3 data_generator.py

# Train the logit-change based forecasting model
echo "Training logit-change based forecasting model..."
python3 forecasting_model.py --model_type logit_change --epochs 5 --batch_size 16 --output models/logit_change_model.pth

# Train the representation-based forecasting model
echo "Training representation-based forecasting model...
python3 forecasting_model.py --model_type representation --epochs 5 --batch_size 16 --output models/representation_model.pth

# Evaluate the models
echo "Evaluating models...
python3 evaluate.py --logit_model models/logit_change_model.pth --representation_model models/representation_model.pth --output output/results.json

# Generate summary report
echo "Generating summary report...
python3 -c "
import json
import datetime

# Load results
with open('output/results.json', 'r') as f:
    results = json.load(f)

# Print summary
print('\n' + '='*60)
print('SUMMARY OF REPRODUCED RESULTS')
print('='*60)
print(f'Reproduction completed at: {datetime.datetime.now()}')
print(f'Logit-change based forecasting F1: {results["logit_change_f1"]:.4f}')
print(f'Representation-based forecasting F1: {results["representation_f1"]:.4f}')
print(f'Edit Success Rate: {results["edit_success_rate"]:.4f}')
print(f'EM Drop Ratio: {results["em_drop_ratio"]:.4f}')
print(f'Performance matches paper? {results["matches_paper']}\n')

# Create a simple report file
with open('output/report.txt', 'w') as f:
    f.write(f'Reproduction Report\n')
    f.write(f'=================\n')
    f.write(f'Reproduction completed at: {datetime.datetime.now()}\n')
    f.write(f'Logit-change based forecasting F1: {results["logit_change_f1']:.4f}\n')
    f.write(f'Representation-based forecasting F1: {results["representation_f1']:.4f}\n')
    f.write(f'Edit Success Rate: {results["edit_success_rate']:.4f}\n')
    f.write(f'EM Drop Ratio: {results["em_drop_ratio']:.4f}\n')
    f.write(f'Performance matches paper? {results["matches_paper']}\n')

echo "Reproduction complete! Results saved to output/ directory"