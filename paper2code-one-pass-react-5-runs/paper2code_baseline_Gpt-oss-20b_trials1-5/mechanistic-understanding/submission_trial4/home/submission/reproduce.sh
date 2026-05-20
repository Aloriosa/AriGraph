#!/usr/bin/env bash
set -e

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo "Extracting toxicity probe vector..."
python src/extract_probe.py

echo "Extracting toxic value vectors..."
python src/extract_toxic_vectors.py

echo "Generating intervention examples..."
python src/intervene_generate.py

echo "Training DPO model..."
python src/train_dpo.py

echo "Evaluating model..."
python src/evaluate.py

echo "Reproduction finished. Results are in the 'results/' folder."