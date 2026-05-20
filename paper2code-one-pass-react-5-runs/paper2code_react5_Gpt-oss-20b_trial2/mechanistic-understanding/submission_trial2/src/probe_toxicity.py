#!/usr/bin/env python
"""
Train a linear probe on the residual stream of the last transformer layer
to predict the toxic / non‑toxic label from the Jigsaw Toxic‑Comment
dataset.  The probe weights are saved to `results/probe.pt`.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from src import config, utils

# -------------------------------------------------------------
# 1. Load models
# -------------------------------------------------------------
print("Loading GPT‑2 medium...")
gpt2, tokenizer = utils.load_model_and_tokenizer(config.MODEL_NAMES["gpt2"])

# -------------------------------------------------------------
# 2. Load Jigsaw dataset
# -------------------------------------------------------------
print("Loading Jigsaw toxic‑comment dataset...")
jigsaw = load_dataset("jigsaw-toxic-comments", split="train")
jigsaw = jigsaw.shuffle(seed=config.SEED).select(range(20000))  # small subset for demo

# -------------------------------------------------------------
# 3. Prepare data
# -------------------------------------------------------------
def get_residuals(texts):
    """Return averaged residual of the last layer for each text."""
    residuals = []
    labels = []
    for text, label in tqdm(zip(texts, labels_list), desc="Encoding"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LENGTH)
        enc = {k: v.to(next(gpt2.parameters()).device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = gpt2(**enc, output_hidden_states=True)
        # hidden_states[-1] is the last layer; shape (batch, seq_len, hidden)
        hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden)
        avg_hidden = hidden.mean(dim=1).squeeze(0)  # (hidden)
        residuals.append(avg_hidden.cpu().numpy())
    return np.stack(residuals)

# Build dataset
texts = [row["comment_text"] for row in jigsaw]
labels_list = [row["toxic"] for row in jigsaw]  # 0/1

print("Computing residuals for the probe dataset...")
residuals = get_residuals(texts)
residuals = torch.tensor(residuals, dtype=torch.float32)
labels = torch.tensor(labels_list, dtype=torch.float32)

# -------------------------------------------------------------
# 4. Train probe
# -------------------------------------------------------------
probe = nn.Linear(gpt2.config.hidden_size, 1).to(residuals.device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(probe.parameters(), lr=config.LEARNING_RATE)

print("Training probe...")
for epoch in range(3):
    perm = torch.randperm(len(residuals))
    for i in tqdm(range(0, len(residuals), config.BATCH_SIZE), desc=f"Epoch {epoch+1}"):
        idx = perm[i:i+config.BATCH_SIZE]
        batch_x = residuals[idx]
        batch_y = labels[idx].unsqueeze(1)
        optimizer.zero_grad()
        logits = probe(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), config.GRAD_CLIP)
        optimizer.step()
print("Probe training complete.")

# -------------------------------------------------------------
# 5. Save probe
# -------------------------------------------------------------
utils.ensure_dir("results")
torch.save(probe.state_dict(), "results/probe.pt")
print("Probe weights saved to results/probe.pt")