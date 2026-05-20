#!/usr/bin/env bash
# Reproduction script for the Refined Coreset Selection (RCS) paper.
#
# This script installs the required Python packages, then runs the
# lightweight implementation in main.py.  The default settings are
# chosen to keep the runtime short while still demonstrating the
# algorithmic skeleton.
#
# Usage:
#   ./reproduce.sh
#
# The script produces a `results.csv` file with the test accuracy,
# selected coreset size and runtime for each dataset.

set -e

# Install dependencies (PyTorch and torchvision).  The container
# already contains CUDA drivers, so the wheel built for CUDA 12.1
# is installed.  If this fails, remove the index-url argument
# and rely on the default PyPI packages.
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Run the reproduction script with default arguments.
python main.py --datasets fashionmnist svhn cifar10 --k 400 --outer_iters 10 --inner_epochs 2