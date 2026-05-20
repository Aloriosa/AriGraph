#!/bin/bash

# Set up environment
cd /home/submission

# Install required packages
apt-get update && apt-get install -y python3 python3-pip python3-venv git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio numpy matplotlib scikit-learn tqdm

# Download and extract the benchmark data (simulated data for the benchmarks)
mkdir -p data
wget -O data/benchmark_data.zip https://github.com/jacksimons15327/snpse_icml/releases/download/v1.0/benchmark_data.zip
unzip -o data/benchmark_data.zip -d data/

# Download the trained models (if available)
mkdir -p models
wget -O models/npse_model.pth https://github.com/jacksimons15327/snpse_icml/releases/download/v1.0/npse_model.pth

# Run the reproduction script
python3 main.py --mode reproduce --num_rounds 10 --simulation_budget 10000 --output_dir results

# Check if the output files were generated
if [ -f "results/output.csv" ]; then
    echo "Reproduction completed successfully. Output saved to results/output.csv"
else
    echo "Error: Output file not found. Reproduction failed."
    exit 1
fi

# Generate summary statistics
python3 summary.py --input results/output.csv --output results/summary.txt

echo "Reproduction script completed. Check results/ directory for output files."