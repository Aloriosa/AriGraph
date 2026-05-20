#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
echo "Installing Python dependencies..."
pip install -q -U pip
pip install -q -r requirements.txt

# Download the black‑box LLM (distilgpt2) and adapter tokenizer
echo "Downloading models..."
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoModelForCausalLM.from_pretrained('distilgpt2'); \
AutoTokenizer.from_pretrained('distilgpt2')"

python -c "from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('bert-base-cased'); \
AutoTokenizer.from_pretrained('bert-base-cased')"

# Create output directory
mkdir -p outputs

# Train the adapter
echo "Starting training..."
python src/train.py \
  --train_file data/train.jsonl \
  --output_dir outputs \
  --epochs 3 \
  --batch_size 4

# Evaluate
echo "Starting evaluation..."
python src/eval.py \
  --test_file data/test.jsonl \
  --adapter_path outputs/checkpoint-epoch-3 \
  --blackbox_name distilgpt2 \
  --output_file outputs/predictions.json

echo "Reproduction completed. Results stored in outputs/predictions.json"