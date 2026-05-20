#!/usr/bin/env bash
set -e

# Install dependencies
pip install -q -U pip
pip install -q -r requirements.txt

# Download the offline dataset (only once)
python -c "import d4rl; print('Dataset download ready')"

# Train FRE encoder and evaluate
python src/train.py \
    --env antmaze-large-diverse-v2 \
    --batch-size 512 \
    --encoder-steps 10000 \
    --output-dir results

# Print results
echo "=== Evaluation Results ==="
cat results.txt