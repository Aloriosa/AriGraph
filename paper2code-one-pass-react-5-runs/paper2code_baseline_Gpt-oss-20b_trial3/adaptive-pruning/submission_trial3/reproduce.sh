#!/usr/bin/env bash
set -e

# 1. Create a virtual environment (optional, but keeps things tidy)
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the experiment
python src/run_experiment.py

# 5. Deactivate virtual environment
deactivate

echo "Reproduction finished successfully."