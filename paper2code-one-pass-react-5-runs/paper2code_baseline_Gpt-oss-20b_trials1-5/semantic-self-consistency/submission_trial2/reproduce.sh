#!/usr/bin/env bash
set -euo pipefail

# Install system packages
apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    python3.10 \
    python3-pip \
    python3-venv

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r requirements.txt

# Run the experiment
python src/main.py