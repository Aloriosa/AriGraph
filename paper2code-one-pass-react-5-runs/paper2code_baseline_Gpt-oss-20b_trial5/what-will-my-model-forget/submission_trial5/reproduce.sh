#!/usr/bin/env bash
# ------------------------------------------------------------
# Reproduce the results of “What Will My Model Forget?”.
# ------------------------------------------------------------
set -euo pipefail

# 1. Install dependencies (Python 3.10+ is required)
echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Download the base language models
echo "Downloading base models..."
python -c "
import transformers
models = ['facebook/bart-base', 'google/flan-t5-base']
for m in models:
    print('Downloading', m)
    transformers.AutoModelForSeq2SeqLM.from_pretrained(m)
    transformers.AutoTokenizer.from_pretrained(m)
"

# 3. Run the experiment script
echo "Running experiment..."
python run_experiment.py

echo "Reproduction finished."