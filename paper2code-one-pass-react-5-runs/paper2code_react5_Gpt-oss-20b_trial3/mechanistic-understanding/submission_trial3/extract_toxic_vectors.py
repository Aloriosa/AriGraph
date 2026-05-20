"""
Extract value vectors (columns of the MLP projection matrix) that
have the highest cosine similarity with the trained probe weight
vector for the toxic class.

The original paper extracts N² ≈ 128² ≈ 16 384 toxic vectors.
Here we extract all MLP value vectors from GPT‑2‑medium,
sort them by cosine similarity with the probe, and keep the top
16 384 vectors.  The list is saved as a list of tuples:
(block_index, unit_index, vector_tensor).
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Model
import os

# Number of toxic vectors to keep (≈128²)
MAX_TOXIC = 128 * 128  # 16384

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load probe and pick the toxic weight (index 1)
probe_state = torch.load("probe.pt", map_location=device)
W_toxic = probe_state["weight"][1]  # shape (hidden,)
W_toxic = F.normalize(W_toxic, dim=0)   # unit vector

# Load GPT‑2‑medium
model = GPT2Model.from_pretrained("gpt2-medium")
model.to(device)
model.eval()

vectors = []

# GPT‑2 block has mlp.c_proj: (hidden, intermediate)
# Each column is a value vector
for block_idx, block in enumerate(model.transformer.h):
    proj_weight = block.mlp.c_proj.weight.data   # (hidden, intermediate)
    for i in range(proj_weight.shape[1]):
        v = proj_weight[:, i]
        v_norm = F.normalize(v, dim=0)
        cos = torch.dot(v_norm, W_toxic).item()
        vectors.append((block_idx, i, v.clone(), cos))

# Sort by cosine similarity (descending)
vectors.sort(key=lambda x: x[3], reverse=True)

# Keep top MAX_TOXIC vectors
top_vectors = [(b, i, v) for b, i, v, c in vectors[:MAX_TOXIC]]
print(f"Extracted top {len(top_vectors)} toxic vectors (max {MAX_TOXIC}).")

# Persist
torch.save(top_vectors, "toxic_vectors.pt")