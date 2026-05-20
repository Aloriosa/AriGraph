#!/usr/bin/env python
"""
Extract the top‑10 most “toxic” value vectors from each GPT‑2 MLP layer
by cosine similarity to the probe vector.
"""
import argparse
import os
import torch
import numpy as np
from transformers import AutoModel
from config import *
from utils import set_all_seeds, get_device

def cosine_similarity(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a @ b.T).squeeze(-1)

def main(args):
    set_all_seeds()
    device = get_device()
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    # Load probe vector
    probe_state = torch.load(args.probe_path, map_location=device)
    probe_linear = torch.nn.Linear(PROBE_HIDDEN_DIM, 1).to(device)
    probe_linear.load_state_dict(probe_state)
    probe_vector = probe_linear.weight.squeeze(0)  # (hidden,)

    toxic_vectors = {}
    # GPT‑2 MLP structure: Linear -> GeLU -> Linear
    # The second Linear's weight (W2) contains value vectors as columns
    for name, module in model.named_modules():
        if "mlp" in name and hasattr(module, "weight"):
            # module.weight shape: (hidden, intermediate * 2) in GPT‑2
            # We need the second part (value vectors)
            w = module.weight.data  # (hidden, hidden * 2)
            hidden, hidden2 = w.shape
            # Split into W1 (hidden, hidden) and W2 (hidden, hidden)
            W1 = w[:, :hidden]
            W2 = w[:, hidden:]  # value vectors
            # Cosine similarity of each column of W2 to probe
            sims = cosine_similarity(W2.t(), probe_vector)  # (hidden,)
            topk = torch.topk(sims, k=10)
            toxic_vectors[name] = {
                "indices": topk.indices.cpu().numpy(),
                "scores": topk.values.cpu().numpy(),
                "vectors": W2[:, topk.indices].cpu().numpy(),
            }
            print(f"[Vectors] Layer {name}: top {len(topk.indices)} toxic vectors extracted.")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(toxic_vectors, os.path.join(args.output_dir, "toxic_vectors.pt"))
    print(f"[Vectors] Extraction finished. Saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=BASE_MODEL)
    parser.add_argument("--probe_path", required=True)
    parser.add_argument("--output_dir", default="output/vectors")
    args = parser.parse_args()
    main(args)