#!/usr/bin/env bash
set -euo pipefail

# Make sure we are in the repository root
cd "$(dirname "$0")"

# Install dependencies
python3 -m pip install -q numpy scipy

# Run the reproduction script
python3 reproduce.py

echo "Reproduction finished."