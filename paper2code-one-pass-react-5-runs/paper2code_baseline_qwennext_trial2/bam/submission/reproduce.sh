#!/bin/bash
# Reproduction script for "Batch and match: black-box variational inference with a score-based divergence"

set -e

# Install required packages
apt-get update
apt-get install -y python3 python3-pip python3-numpy python3-scipy python3-matplotlib python3-jax python3-jaxlib

# Create directory for outputs
mkdir -p outputs

# Run the BaM algorithm implementation
python3 bam_vi.py

# Create summary of results
python3 summarize_results.py

echo "Reproduction complete. Results saved in outputs/ directory"