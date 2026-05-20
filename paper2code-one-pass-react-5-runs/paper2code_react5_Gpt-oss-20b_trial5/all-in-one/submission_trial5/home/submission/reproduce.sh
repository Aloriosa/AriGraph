#!/usr/bin/env bash
set -euo pipefail

# 1. Update the package index and install system dependencies
apt-get update && apt-get install -y python3-pip git

# 2. Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the simulation–based inference demo
python run_simulation.py

# 5. Inform the user that the output has been saved
echo "Simulation-based inference demo finished. Results are in the 'output' directory."