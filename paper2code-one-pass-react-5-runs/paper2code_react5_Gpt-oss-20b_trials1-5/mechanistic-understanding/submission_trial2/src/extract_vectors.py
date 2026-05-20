#!/usr/bin/env python
"""
Extract key and value vectors that are most aligned with the toxic probe
and perform SVD on the selected vectors.  Results are printed to stdout
and saved to `results/`.  The output can be inspected manually.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM
from src import config, utils

# -------------------------------------------------------------
# Load model and probe
# -------------------------------------------------------------
print("Loading GPT‑2 medium...")
gpt2, tokenizer = utils.load_model_and_tokenizer(config.MODEL_NAMES["gpt2"])
print("Loading probe weights...")
probe = torch.nn.Linear(gpt2.config.hidden_size, 1)
probe.load_state_dict(torch.load("results/probe.pt", map_location="cpu"))
probe.eval()
probe_vec = probe.weight.squeeze(0).cpu()  # (hidden)

# -------------------------------------------------------------
# Utility: cosine similarity
# -------------------------------------------------------------
def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# -------------------------------------------------------------
# 1. Extract key vectors (rows of the first MLP weight)
# -------------------------------------------------------------
top_k = 5
key_vectors = []
for i, layer in enumerate(gpt2.transformer.h):
    # key weight: c_fc.weight (hidden, inner)
    weight = layer.mlp.c_fc.weight.detach().cpu()  # (hidden, inner)
    for j in range(weight.shape[1]):  # each column is a key vector
        vec = weight[:, j]
        score = cosine_sim(vec, probe_vec)
        key_vectors.append((score, i, j, vec))
key_vectors.sort(reverse=True, key=lambda x: x[0])
print(f"Top {top_k} key vectors most aligned with probe:")

for idx in range(top_k):
    score, layer, col, vec = key_vectors[idx]
    print(f"  Layer {layer}, key column {col}, score={score:.4f}")

# -------------------------------------------------------------
# 2. Extract value vectors (columns of the second MLP weight)
# -------------------------------------------------------------
value_vectors = []
for i, layer in enumerate(gpt2.transformer.h):
    weight = layer.mlp.c_proj.weight.detach().cpu()  # (inner, hidden)
    for j in range(weight.shape[0]):  # each row is a value vector
        vec = weight[j, :]
        score = cosine_sim(vec, probe_vec)
        value_vectors.append((score, i, j, vec))
value_vectors.sort(reverse=True, key=lambda x: x[0])
print(f"\nTop {top_k} value vectors most aligned with probe:")

for idx in range(top_k):
    score, layer, row, vec = value_vectors[idx]
    print(f"  Layer {layer}, value row {row}, score={score:.4f}")

# -------------------------------------------------------------
# 3. SVD on the top value vectors
# -------------------------------------------------------------
svd_input = torch.stack([vec for _, _, _, vec in value_vectors[:top_k]])  # (k, hidden)
U, S, Vh = torch.linalg.svd(svd_input, full_matrices=False)
print("\nSVD of top value vectors:")
for i in range(min(3, U.shape[0])):
    print(f"  Singular vector {i}, singular value {S[i].item():.4f}")

# -------------------------------------------------------------
# 4. Save results
# -------------------------------------------------------------
utils.ensure_dir("results")
np.save("results/key_vectors.npy", np.array([vec.numpy() for _, _, _, vec in key_vectors[:top_k]]))
np.save("results/value_vectors.npy", np.array([vec.numpy() for _, _, _, vec in value_vectors[:top_k]]))
np.save("results/svd_U.npy", U.cpu().numpy())
np.save("results/svd_S.npy", S.cpu().numpy())
np.save("results/svd_Vh.npy", Vh.cpu().numpy())
print("\nSaved extracted vectors and SVD to results/ directory.")