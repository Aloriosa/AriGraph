#!/bin/bash
# Reproduction script for "Batch and match: black-box variational inference with a score-based divergence"

# Set up environment
set -e  # Exit on any error

echo "Setting up environment for reproduction..."

# Install Python and required packages
apt-get update
apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv /tmp/reproduction_env
source /tmp/reproduction_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install numpy scipy matplotlib jax jaxlib

# Verify installations
echo "Python version: $(python --version)"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"
echo "SciPy version: $(python -c 'import scipy; print(scipy.__version__)')"
echo "JAX version: $(python -c 'import jax; print(jax.__version__)')"

echo "Environment setup complete."

# Create output directory
mkdir -p /home/submission/output

# Run the main reproduction script
echo "Running Batch and Match (BaM) algorithm..."
python3 batch_and_match.py --iterations 100 --batch-size 32 --output /home/submission/output/results.pkl

echo "Batch and Match (BaM) algorithm completed."

# Run comparison with ADVI and GSM
echo "Running ADVI comparison..."
python3 compare_algorithms.py --iterations 100 --batch-size 32 --output /home/submission/output/comparison_results.pkl

echo "All experiments completed."

# Generate summary statistics
echo "Generating summary statistics..."
python3 generate_summary.py --input /home/submission/output/results.pkl --output /home/submission/output/summary.txt

echo "Reproduction complete. Results saved to /home/submission/output/"

# Print summary
echo "Summary of results:"
cat /home/submission/output/summary.txt