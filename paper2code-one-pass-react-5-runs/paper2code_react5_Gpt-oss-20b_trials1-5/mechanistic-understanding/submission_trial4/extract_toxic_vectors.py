import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import set_seed, ensure_dir, load_tokenizer, load_model
import yaml

config = yaml.safe_load(open("config.yaml"))
set_seed(config["seed"])
device = config["device"]
probe_cfg = config["probe"]

probe_path = os.path.join(probe_cfg["output_path"], "probe.pt")
probe_vector = torch.load(probe_path, map_location=device)
probe_vector = probe_vector.squeeze(0)  # shape: hidden_size

model_name = probe_cfg["model_name"]
tokenizer = load_tokenizer(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# Get transformer blocks
blocks = [block for block in model.transformer.h]
num_layers = len(blocks)
hidden_size = model.config.hidden_size
mlp_dim = model.config.intermediate_size

# Collect toxic value vectors
top_n = 128
all_vectors = []
all_keys = []

with torch.no_grad():
    for l, block in enumerate(blocks):
        # block.mlp has two linear layers: fc1 (W_k) and fc2 (W_v)
        W_k = block.mlp.c_fc.weight  # shape: (mlp_dim, hidden_size)
        W_v = block.mlp.c_proj.weight  # shape: (hidden_size, mlp_dim)
        # keys: rows of W_k
        keys = W_k.detach()  # (mlp_dim, hidden_size)
        # values: columns of W_v transposed
        values = W_v.t().detach()  # (mlp_dim, hidden_size)
        # compute cosine similarity with probe
        cos = torch.nn.functional.cosine_similarity(values, probe_vector.unsqueeze(0), dim=1)
        topk_idx = torch.topk(cos, top_n, largest=True).indices
        top_vals = values[topk_idx]
        top_keys = keys[topk_idx]
        all_vectors.append(top_vals.cpu().numpy())
        all_keys.append(top_keys.cpu().numpy())

all_vectors = np.concatenate(all_vectors, axis=0)  # (num_layers*top_n, hidden_size)

# SVD
U, S, Vh = np.linalg.svd(all_vectors, full_matrices=False)
# Take first 3 singular vectors
svd_vectors = U[:, :3]  # shape: (num_vectors, hidden_size)

# Project onto vocab space
embed_matrix = tokenizer.get_input_embeddings().weight.detach().cpu().numpy()  # (vocab_size, hidden_size)
top_tokens = []
for vec in svd_vectors:
    proj = embed_matrix @ vec
    top_k = np.argsort(-proj)[:10]
    tokens = [tokenizer.decode([i]) for i in top_k]
    top_tokens.append(tokens)

ensure_dir("outputs/vectors")
np.save("outputs/vectors/svd_vectors.npy", svd_vectors)
np.save("outputs/vectors/top_tokens.npy", np.array(top_tokens))
print("Toxic vectors extracted and saved to outputs/vectors")