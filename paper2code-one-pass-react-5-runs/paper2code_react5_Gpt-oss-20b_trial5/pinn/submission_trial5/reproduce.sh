#!/usr/bin/bash
set -e

# 1. Create a virtual environment (Python 3.10)
python3 -m venv venv
source venv/bin/activate

# 2. Install required packages
pip install --upgrade pip
pip install torch==2.0.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm jsonlines matplotlib

# 3. Run the full experiment pipeline
python experiments/main.py

# 4. Deactivate the virtual environment
deactivate