#!/usr/bin/bash
set -e

# Install dependencies
pip install --quiet -r requirements.txt

# Run the LBCS core selection
python -m lbcslite.main --dataset mnist --k 1000 --epsilon 0.2 --iterations 100 --inner_epochs 5

# Train a model on the selected coreset and evaluate on test set
python -m lbcslite.train_coreset --dataset mnist --mask_file output/selected_mask.npy --inner_epochs 10

echo "Reproduction finished. Results are in the 'output' directory."