#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy matplotlib tqdm scikit-learn

# Run the reproduction script
python3 main.py --task inpainting --epochs 50 --batch_size 32 --learning_rate 2e-4 --output results_inpainting
python3 main.py --task super_resolution --epochs 50 --batch_size 32 --learning_rate 2e-4 --output results_super_resolution

# Generate summary report
python3 evaluate.py --input results_inpainting --output summary_inpainting.csv
python3 evaluate.py --input results_super_resolution --output summary_super_resolution.csv

echo "Reproduction complete. Results saved in results_inpainting/ and results_super_resolution/ directories."