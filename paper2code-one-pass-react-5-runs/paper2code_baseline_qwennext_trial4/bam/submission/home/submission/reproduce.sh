#!/bin/bash

# Set up the environment
apt-get update && apt-get install -y python3 python3-pip python3-numpy python3-scipy python3-matplotlib python3-jax python3-jaxlib

# Install required packages
pip3 install jax jaxlib numpy scipy matplotlib

# Create directory for outputs
mkdir -p /home/submission/output

# Run the BaM algorithm on the Gaussian target
python3 /home/submission/bam.py --target gaussian --dim 16 --batch-size 10 --iterations 100 --output /home/submission/output/gaussian_results.pkl

# Run BaM on non-Gaussian target
python3 /home/submission/bam.py --target sinh_arcsinh --dim 10 --batch-size 10 --iterations 100 --skew 1.0 --tail 1.0 --output /home/submission/output/non_gaussian_results.pkl

# Run comparisons against baselines
python3 /home/submission/compare_baselines.py --target gaussian --dim 16 --batch-size 10 --iterations 100 --output /home/submission/output/comparison_results.pkl

# Generate plots
python3 /home/submission/plot_results.py --input /home/submission/output/gaussian_results.pkl --output /home/submission/output/gaussian_plot.png

python3 /home/submission/plot_results.py --input /home/submission/output/non_gaussian_results.pkl --output /home/submission/output/non_gaussian_plot.png

python3 /home/submission/plot_results.py --input /home/submission/output/comparison_results.pkl --output /submission/output/comparison_plot.png

# Print completion message
echo "Reproduction complete. Results saved to /home/submission/output/"