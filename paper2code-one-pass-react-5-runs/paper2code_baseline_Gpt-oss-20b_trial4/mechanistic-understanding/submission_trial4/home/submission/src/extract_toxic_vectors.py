import os
import json
import numpy as np
import torch
from transformers import GPT2Model, GPT2TokenizerFast

def main():
    model_name = "gpt2-medium"
    probe_path = "results/probe_vector.npy"
    probe_vector = np.load(probe_path)

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Token embeddings (vocab x d)
    embed = model.wte.weight.data.cpu().numpy()  # (vocab, d)

    toxic_vectors = []
    for layer_idx, block in enumerate(model.transformer.h):
        # c_proj weight shape (d, 4*d) – each column is a value vector
        c_proj = block.mlp.c_proj.weight.data.cpu().numpy()  # (d, 4*d)
        d_mlp = c_proj.shape[1]
        for i in range(d_mlp):
            val_vec = c_proj[:, i]
            cos = np.dot(val_vec, probe_vector) / (
                np.linalg.norm(val_vec) * np.linalg.norm(probe_vector) + 1e-8
            )
            toxic_vectors.append((cos, layer_idx, i, val_vec))

    toxic_vectors.sort(key=lambda x: x[0], reverse=True)
    top_k = 5
    results = []
    for cos, layer, idx, vec in toxic_vectors[:top_k]:
        logits = embed @ vec  # (vocab,)
        top_tokens = np.argsort(logits)[-10:][::-1]
        tokens = [tokenizer.decode([t]) for t in top_tokens]
        results.append(
            {
                "layer": layer,
                "index": idx,
                "cosine": float(cos),
                "top_tokens": tokens,
            }
        )

    os.makedirs("results", exist_ok=True)
    with open("results/toxic_vectors.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Top toxic vectors saved to results/toxic_vectors.json")

if __name__ == "__main__":
    main()