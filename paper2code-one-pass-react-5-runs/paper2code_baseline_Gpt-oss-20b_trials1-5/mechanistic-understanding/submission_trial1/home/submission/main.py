#!/usr/bin/env python
"""
Entry point for the reproduction pipeline.
"""

import os
import json
import random
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from probe import train_probe
from intervene import intervene_and_generate
from dpo_train import train_dpo
from metrics import compute_toxicity, evaluate_model

# ------------------------------
# Configuration
# ------------------------------
MODEL_ID = "gpt2-medium"            # Change to "meta-llama/Llama-2-7b-hf" if GPU memory allows
PROBE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DPO_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_PROMPTS = 100          # Number of prompts to evaluate
INTERVENTION_ALPHA = 0.3   # Scaling factor for residual‑stream intervention
DPO_EPOCHS = 1
DPO_BATCH = 4
DPO_LEARNING_RATE = 1e-6
DPO_BETA = 0.1
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------
# Load tokenizer & base model
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # GPT‑2 has no pad token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
base_model.eval()
base_model.to(PROBE_DEVICE)

# ------------------------------
# 1. Train linear probe
# ------------------------------
print("\n=== 1. Training linear toxicity probe ...")
probe_weights = train_probe(base_model, tokenizer, device=PROBE_DEVICE,
                            epochs=3, batch_size=8)
torch.save(probe_weights, os.path.join(OUTPUT_DIR, "model_probe.pt"))
print("Probe weights saved.")

# ------------------------------
# 2. Extract top toxic MLP vectors
# ------------------------------
print("\n=== 2. Extracting top toxic MLP value vectors ...")
top_vectors = []  # list of (layer_idx, vector_idx, vector)
for layer_idx, layer in enumerate(base_model.transformer.h):
    # GPT‑2 MLP value vectors are columns of c_proj.weight
    value_vectors = layer.mlp.c_proj.weight.T  # shape: (hidden_dim, hidden_dim*4)
    # Compute cosine similarity with probe
    sims = torch.nn.functional.cosine_similarity(value_vectors, probe_weights.unsqueeze(0),
                                                 dim=1)
    topk = torch.topk(sims, k=3).indices.tolist()
    for v_idx in topk:
        top_vectors.append((layer_idx, v_idx, value_vectors[:, v_idx].cpu()))

print(f"Selected {len(top_vectors)} toxic vectors (layer, idx).")

# ------------------------------
# 3. Prepare evaluation prompts
# ------------------------------
print("\n=== 3. Loading evaluation prompts ...")
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
prompts = [x.replace("\n", " ") for x in wikitext["text"][:NUM_PROMPTS]]
print(f"Using {len(prompts)} prompts.")

# ------------------------------
# 4. Baseline generation & toxicity
# ------------------------------
print("\n=== 4. Baseline toxicity evaluation ...")
baseline_outputs = []
for p in prompts:
    out = base_model.generate(
        tokenizer.encode(p, return_tensors="pt").to(PROBE_DEVICE),
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    baseline_outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))
baseline_tox = compute_toxicity(baseline_outputs)
print(f"Baseline toxicity score: {baseline_tox:.4f}")

# ------------------------------
# 5. Intervention evaluation
# ------------------------------
print("\n=== 5. Intervention evaluation ...")
interv_outputs = []
for p in prompts:
    out = intervene_and_generate(
        base_model, tokenizer, p,
        alpha=INTERVENTION_ALPHA,
        device=PROBE_DEVICE,
        max_new_tokens=20
    )
    interv_outputs.append(out)
interv_tox = compute_toxicity(interv_outputs)
print(f"After intervention toxicity score: {interv_tox:.4f}")

# ------------------------------
# 6. DPO fine‑tuning
# ------------------------------
print("\n=== 6. DPO fine‑tuning ...")
# Create synthetic pairwise dataset
pairwise_ds = []
for p in prompts:
    # Positive: greedy
    pos_ids = base_model.generate(
        tokenizer.encode(p, return_tensors="pt").to(PROBE_DEVICE),
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    pos = tokenizer.decode(pos_ids[0], skip_special_tokens=True)

    # Negative: sampling with temperature 1.0
    neg_ids = base_model.generate(
        tokenizer.encode(p, return_tensors="pt").to(PROBE_DEVICE),
        max_new_tokens=20,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    neg = tokenizer.decode(neg_ids[0], skip_special_tokens=True)

    pairwise_ds.append({"prompt": p, "chosen": pos, "rejected": neg})

# Train DPO
dpo_model = train_dpo(pairwise_ds, tokenizer, device=DPO_DEVICE,
                      epochs=DPO_EPOCHS, batch_size=DPO_BATCH,
                      learning_rate=DPO_LEARNING_RATE, beta=DPO_BETA)
torch.save(dpo_model.state_dict(), os.path.join(OUTPUT_DIR, "model_dpo.pt"))
print("DPO model saved.")

# ------------------------------
# 7. DPO evaluation
# ------------------------------
print("\n=== 7. DPO evaluation ...")
dpo_outputs = []
for p in prompts:
    out = dpo_model.generate(
        tokenizer.encode(p, return_tensors="pt").to(DPO_DEVICE),
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    dpo_outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))
dpo_tox = compute_toxicity(dpo_outputs)
print(f"After DPO toxicity score: {dpo_tox:.4f}")

# ------------------------------
# 8. Save final metrics
# ------------------------------
metrics = {
    "baseline_toxicity": baseline_tox,
    "intervention_toxicity": interv_tox,
    "dpo_toxicity": dpo_tox
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== Reproduction finished. Metrics saved to outputs/metrics.json ===")