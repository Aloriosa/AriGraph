#!/usr/bin/env python3
"""
Train a simple linear probe that maps the averaged last‑layer residual
(from GPT‑2‑medium) to a binary toxicity label.
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class LinearProbe(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # (batch,)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, output_hidden_states=True
    ).to(device)
    model.eval()

    # Load Jigsaw Toxic comment dataset (train split)
    dataset = load_dataset("jigsaw-toxic-comment-classification-challenge", split="train")
    # We use only the first 5000 examples for speed
    dataset = dataset.select(range(5000))
    texts = dataset["text"]
    labels = np.array(dataset["toxic"], dtype=np.float32)

    # Tokenize and obtain residuals
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # hidden_states is a tuple: (embeddings, h1, h2, ..., h12)
        hidden_states = outputs.hidden_states  # tuple of tensors
        last_hidden = hidden_states[-1]  # (batch, seq_len, hidden)
        # Average over tokens
        residual = last_hidden.mean(dim=1)  # (batch, hidden)

    # Prepare dataloader
    dataset_tensor = TensorDataset(residual.cpu(), torch.tensor(labels))
    dataloader = DataLoader(dataset_tensor, batch_size=32, shuffle=True)

    # Train probe
    probe = LinearProbe().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(probe.parameters(), lr=1e-3)

    epochs = 3
    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = probe(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    # Save probe weights (we only need the linear layer weight)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(probe.state_dict(), os.path.join(args.output_dir, "probe.pt"))
    print(f"Probe weights saved to {os.path.join(args.output_dir, 'probe.pt')}")

if __name__ == "__main__":
    main()