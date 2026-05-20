#!/bin/bash

# Set up environment
apt-get update && apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy matplotlib scipy scikit-learn tqdm

# Create directories
mkdir -p /home/submission/results
mkdir -p /home/submission/data

# Run BAM algorithm on Gaussian target distribution
python3 bam_algorithm.py --target gaussian --dim 10 --batch_size 100 --max_iterations 50 --regularization 1e-6 --output results/gaussian_results.pkl

# Run BAM algorithm on non-Gaussian target (mixture of Gaussians)
python3 bam_algorithm.py --target mixture --dim 10 --batch_size 100 --max_iterations 50 --regularization 1e-6 --output results/mixture_results.pkl

# Run BAM algorithm on hierarchical model (linear regression)
python3 bam_algorithm.py --target hierarchical --dim 10 --batch_size 100 --max_iterations 50 --regularization 1e-6 --output results/hierarchical_results.pkl

# Generate convergence plots and summary statistics
python3 evaluate_bam.py --input results/gaussian_results.pkl --target gaussian --output results/gaussian_summary.csv
python3 evaluate_bam.py --input results/mixture_results.pkl --target mixture --output results/mixture_summary.csv
python3 evaluate_bam.py --input results/hierarchical_results.pkl --target hierarchical --output results/hierarchical_summary.csv

# Generate comparison with BBVI baseline
python3 compare_baselines.py --output results/baseline_comparison.csv

# Generate final summary report
python3 generate_report.py --input results/gaussian_summary.csv --input results/mixture_summary.csv --input results/hierarchical_summary.csv --output results/final_report.txt

echo "Reproduction complete! Results saved in results/ directory."