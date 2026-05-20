#!/usr/bin/env python
"""
Project the extracted value vectors into vocabulary space and print the
top tokens most promoted by each vector.  The embeddings are taken from
the model's word‑embedding matrix.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from src import config, utils

# -------------------------------------------------------------
# Load model and embeddings
# -------------------------------------------------------------
print("Loading GPT‑2 medium...")
gpt2, tokenizer = utils.load_model_and_tokenizer(config.MODEL_NAMES["gpt2"])
embedding = gpt2.get_input_embeddings().weight.detach().cpu()  # (vocab, hidden)

# -------------------------------------------------------------
# Load value vectors
# -------------------------------------------------------------
values = torch.tensor(np.load("results/value_vectors.npy"))  # (k, hidden)
k = values.shape[0]
print(f"Projecting {k} value vectors into vocab space...")

for i in range(k):
    vec = values[i]
    # projection: vocab x hidden -> (vocab,)
    proj = embedding @ vec  # (vocab,)
    top_ids = torch.topk(proj, 10).indices.tolist()
    top_tokens = [tokenizer.decode([tid]) for tid in top_ids]
    print(f"Vector {i}: top tokens: {top_tokens}")

print("\nProjection complete.  Results are printed above.")