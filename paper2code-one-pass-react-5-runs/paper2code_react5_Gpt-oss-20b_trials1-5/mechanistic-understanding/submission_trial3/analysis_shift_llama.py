"""
Compute and print statistics of weight changes between the original
LLaMA‑2‑7b and the DPO‑fine‑tuned version.  For every weight matrix
we report the cosine similarity and L2 norm difference.
The results are written to shift_stats_llama.txt.
"""

import torch
import numpy as np
from transformers import LlamaForCausalLM

# Load models
orig = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
dpo = LlamaForCausalLM.from_pretrained("llama2_dpo")  # optional DPO checkpoint

orig_state = orig.state_dict()
dpo_state = dpo.state_dict()

stats = []

for k in orig_state:
    if k not in dpo_state:
        continue
    w_orig = orig_state[k].flatten()
    w_dpo = dpo_state[k].flatten()
    # Cosine similarity
    cos = torch.dot(w_orig, w_dpo).item() / (
        w_orig.norm().item() * w_dpo.norm().item()
    )
    # L2 norm difference
    diff = (w_orig - w_dpo).norm().item()
    stats.append((k, cos, diff))

# Write summary
with open("shift_stats_llama.txt", "w") as f:
    f.write(f"{'Param':>60}  {'Cosine':>10}  {'L2 Diff':>10}\n")
    for k, cos, diff in stats:
        f.write(f"{k:>60}  {cos:10.4f}  {diff:10.4f}\n")

# Print overall averages
avg_cos = np.mean([s[1] for s in stats])
avg_diff = np.mean([s[2] for s in stats])
print(f"Average cosine similarity across all params: {avg_cos:.4f}")
print(f"Average L2 norm difference across all params: {avg_diff:.6f}")
print("Detailed stats written to shift_stats_llama.txt")