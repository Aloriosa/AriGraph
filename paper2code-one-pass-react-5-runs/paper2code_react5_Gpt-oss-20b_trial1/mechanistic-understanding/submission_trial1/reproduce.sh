#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# 1. Install system dependencies
# ------------------------------------------------------------
echo "Installing system packages..."
sudo apt-get update -y && sudo apt-get install -y git python3-venv

# ------------------------------------------------------------
# 2. Create a fresh virtual environment
# ------------------------------------------------------------
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# ------------------------------------------------------------
# 3. Install Python dependencies
# ------------------------------------------------------------
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# ------------------------------------------------------------
# 4. Run the full reproduction pipeline
# ------------------------------------------------------------
echo "Running the reproduction pipeline..."
mkdir -p output

# 4.1 Train a linear probe on the Jigsaw toxicity dataset
python src/train_probe.py \
  --model_name gpt2-medium \
  --output_dir output/probe

# 4.2 Extract toxic value vectors from GPT‑2
python src/extract_toxic_vectors.py \
  --model_name gpt2-medium \
  --probe_path output/probe/probe.pt \
  --output_path output/toxic_vectors.json

# 4.3 Generate pairwise (positive/negative) data for DPO
python src/generate_pairwise_data.py \
  --model_name gpt2-medium \
  --probe_path output/probe/probe.pt \
  --output_path output/pairwise_data.json \
  --num_pairs 2000

# 4.4 Fine‑tune GPT‑2 with DPO
python src/dpo_finetune.py \
  --model_name gpt2-medium \
  --pairwise_path output/pairwise_data.json \
  --output_dir output/dpo_model

# 4.5 Evaluate the fine‑tuned model
python src/evaluate.py \
  --model_dir output/dpo_model \
  --output_dir output/eval

# 4.6 Perform a simple intervention (subtract one toxic vector)
python src/intervention.py \
  --model_dir output/dpo_model \
  --toxic_vector_path output/toxic_vectors.json \
  --output_dir output/intervention

# 4.7 Un‑align the model by scaling a key vector
python src/unalign.py \
  --model_dir output/dpo_model \
  --toxic_vector_path output/toxic_vectors.json \
  --output_dir output/unalign

echo "All steps completed. Results are in the 'output' directory."