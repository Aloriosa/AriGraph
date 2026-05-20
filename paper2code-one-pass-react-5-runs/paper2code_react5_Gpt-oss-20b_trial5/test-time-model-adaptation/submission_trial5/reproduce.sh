#!/usr/bin/env bash
set -euo pipefail

# Ensure the script is executed from the repository root
cd "$(dirname "$0")"

# Create a clean Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the FOA reproduction script
python foa.py