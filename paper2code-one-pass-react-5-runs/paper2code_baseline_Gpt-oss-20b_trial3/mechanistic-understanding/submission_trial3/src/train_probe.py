#!/usr/bin/env python
"""
Train a linear probe on the Jigsaw toxic comment dataset.
The probe learns to predict toxicity from the last hidden state of GPT‑2.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from tqdm import tqdm
from config import *
from utils import set_all_seeds, get_device, average_last_hidden, softmax_cross_entropy

def main(args):
    set_all_seeds()
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    # Load Jigsaw dataset
    raw_ds = load_dataset("jigsaw-toxic-comment-classification-challenge", split=JIGSAW_SPLIT)
    # Map to lower‑case and add `label`
    def preprocess(ex):
        ex["label"] = float(ex["toxic"])
        ex["text"] = ex["comment_text"]
        return ex
    ds = raw_ds.map(preprocess, remove_columns=raw_ds.column_names)

    # Tokenize
    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=512)
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    dataloader = DataLoader(ds, batch_size=PROBE_BATCH_SIZE, shuffle=True)

    # Load GPT‑2 base model
    gpt = AutoModel.from_pretrained(BASE_MODEL, output_hidden_states=True).to(device)
    gpt.eval()

    # Probe: linear layer
    probe = torch.nn.Linear(PROBE_HIDDEN_DIM, 1).to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-4)

    for epoch in range(PROBE_EPOCHS):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Probe Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = gpt(**batch)
                last_hidden = average_last_hidden(outputs)  # (batch, hidden)
            logits = probe(last_hidden).squeeze(-1)  # (batch,)
            loss = softmax_cross_entropy(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[Probe] Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

    # Save probe weights
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(probe.state_dict(), os.path.join(args.output_dir, "probe.pt"))
    print(f"[Probe] Training finished. Probe vector saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=BASE_MODEL)
    parser.add_argument("--output_dir", default="output/probe")
    args = parser.parse_args()
    main(args)