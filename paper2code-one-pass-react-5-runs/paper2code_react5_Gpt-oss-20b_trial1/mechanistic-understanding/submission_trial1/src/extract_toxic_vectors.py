#!/usr/bin/env python3
"""
Extract the top‑k MLP value vectors from GPT‑2‑medium whose
cosine similarity with the probe weight is largest.
Also compute an SVD basis of the extracted vectors.
"""

import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load probe weight
    probe_state = torch.load(args.probe_path, map_location="cpu")
    probe_weight = probe_state["linear.weight"].squeeze(0).cpu()  # (hidden,)

    # Load GPT‑2‑medium
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    toxic_vectors = []
    for layer_idx, layer in enumerate(model.transformer.h):
        # GPT‑2 uses mlp.c_proj as the second linear (intermediate -> hidden)
        proj_weight = layer.mlp.c_proj.weight.data.cpu()  # (hidden, intermediate)
        # Each column is a value vector
        for idx in range(proj_weight.shape[1]):
            vec = proj_weight[:, idx]
            cos_sim = F.cosine_similarity(vec, probe_weight, dim=0).item()
            toxic_vectors.append({
                "layer": layer_idx,
                "index": idx,
                "cosine": cos_sim,
                "value": vec.numpy().tolist()
            })

    # Sort by cosine similarity (descending)
    toxic_vectors.sort(key=lambda x: x["cosine"], reverse=True)
    top_k = 128
    top_vectors = toxic_vectors[:top_k]

    # Compute SVD on the matrix of shape (top_k, hidden)
    mat = np.stack([v["value"] for v in top_vectors], axis=0)  # (k, hidden)
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    # Keep U (k, hidden)
    svd_basis = U.tolist()

    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({
            "top_vectors": top_vectors,
            "svd_basis": svd_basis
        }, f, indent=2)

    print(f"Extracted {top_k} toxic vectors and SVD basis to {args.output_path}")

if __name__ == "__main__":
    main()