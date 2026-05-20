#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install -r requirements.txt

# 1. Train linear toxicity probe
python train_probe.py

# 2. Extract toxic vectors
python extract_toxic_vectors.py

# 3. Compute SVD basis of toxic vectors
python svd_toxic_vectors.py

# 4. Generate pairwise dataset (PPLM style)
python prepare_pairs.py

# 5. Fine‑tune GPT‑2 with DPO
python dpo_train.py

# 6. Evaluate baseline, aligned and re‑aligned models
python evaluate.py

# 7. Analyse weight shifts caused by DPO (GPT‑2)
python analysis_shift.py

# 8. Analyse weight shifts caused by DPO (LLaMA‑2‑7b, if available)
python analysis_shift_llama.py

# 9. Demonstrate simple re‑alignment attack
python realign_gpt2.py

# Optional: LLaMA‑2‑7b gating analysis (requires model)
# python llm_gating_analysis.py

# Optional: LLaMA‑2‑7b re‑alignment (requires DPO checkpoint)
# python llm_realign.py