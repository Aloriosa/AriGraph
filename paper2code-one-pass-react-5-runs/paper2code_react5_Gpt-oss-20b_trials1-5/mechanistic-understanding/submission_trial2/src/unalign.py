#!/usr/bin/env python
"""
Re‑activate toxicity in the DPO‑fine‑tuned GPT‑2 model by scaling the
key vectors that were most aligned with the toxic probe.  The scaled
model is saved as `results/unaligned_gpt2`.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from src import config, utils

# Load the DPO model
print("Loading fine‑tuned DPO GPT‑2...")
model, tokenizer = utils.load_model_and_tokenizer("results/dpo_gpt2")

# Load probe vector
probe_vec = torch.load("results/probe.pt", map_location="cpu")["weight"].squeeze(0)

# -------------------------------------------------------------
# Identify key vectors most aligned with probe
# -------------------------------------------------------------
def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

top_k = 7
aligned_keys = []

for i, layer in enumerate(model.transformer.h):
    key_weight = layer.mlp.c_fc.weight.detach()  # (hidden, inner)
    for j in range(key_weight.shape[1]):
        vec = key_weight[:, j]
        score = cosine_sim(vec, probe_vec)
        aligned_keys.append((score, i, j, vec))

aligned_keys.sort(reverse=True, key=lambda x: x[0])
print(f"Scaling top {top_k} key vectors by factor 10 to re‑activate toxicity.")
for idx in range(top_k):
    _, layer_idx, col_idx, _ = aligned_keys[idx]
    layer = model.transformer.h[layer_idx]
    with torch.no_grad():
        layer.mlp.c_fc.weight[:, col_idx] *= 10.0

# -------------------------------------------------------------
# Save un‑aligned model
# -------------------------------------------------------------
utils.ensure_dir("results")
model.save_pretrained("results/unaligned_gpt2")
tokenizer.save_pretrained("results/unaligned_gpt2")
print("Un‑aligned model saved to results/unaligned_gpt2/")