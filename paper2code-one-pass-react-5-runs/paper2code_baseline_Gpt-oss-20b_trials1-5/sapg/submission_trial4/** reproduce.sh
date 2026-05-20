#!/usr/bin/env bash
set -euo pipefail

# 1. Update package lists and install system dependencies
echo "Updating package list..."
apt-get update -qq

# 2. Install Python 3.10+ (Ubuntu 24.04 already has 3.10)
pip install --upgrade pip

# 3. Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 4. Run the SAPG training demo
echo "Starting SAPG training demo..."
python -m src.sapg

# 5. Verify that results.csv exists
if [ ! -f results.csv ]; then
  echo "ERROR: results.csv not found!"
  exit 1
fi

echo "Reproduction script finished successfully."