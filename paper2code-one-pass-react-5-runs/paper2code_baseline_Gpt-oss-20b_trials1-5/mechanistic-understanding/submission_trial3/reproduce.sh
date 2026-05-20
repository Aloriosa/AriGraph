#!/usr/bin/env bash
set -e

# 1. Install dependencies
pip install -r requirements.txt

# 2. Train a linear probe on the Jigsaw toxic comment dataset
python src/train_probe.py --model_name gpt2-medium --output_dir output/probe

# 3. Extract toxic value vectors from GPT2
python src/extract_toxic_vectors.py \
  --model_name gpt2-medium \
  --probe_path output/probe/probe.pt \
  --output_dir output/vectors

# 4. Fine‑tune GPT2 with DPO on a small toxic pairwise dataset
python src/dpo_finetune.py \
  --base_model gpt2-medium \
  --probe_path output/probe/probe.pt \
  --output_dir output/dpo

# 5. Evaluate the baseline, the DPO model and the “un‑aligned” model
python src/evaluate.py \
  --base_model gpt2-medium \
  --dpo_model output/dpo/best_model.pt \
  --probe_path output/probe/probe.pt

echo "Reproduction finished. Check the output/ directory for results."