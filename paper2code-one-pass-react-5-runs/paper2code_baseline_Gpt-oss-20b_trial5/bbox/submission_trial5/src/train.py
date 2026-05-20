#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.adapter import Adapter
from src.utils import QADataset, collate_fn, generate_candidates, set_seed

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_path = Path("data/train.jsonl")
    train_ds = QADataset(train_path)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              collate_fn=collate_fn)

    # ------------------------------------------------------------------
    # 2. Load black‑box GPT‑2
    # ------------------------------------------------------------------
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # required for generation
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt_model.eval()  # keep GPT‑2 frozen

    # ------------------------------------------------------------------
    # 3. Init adapter
    # ------------------------------------------------------------------
    adapter = Adapter("bert-base-uncased").to(device)
    optimizer = torch.optim.AdamW(adapter.scorer.parameters(), lr=5e-5)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    epochs = 5
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_q, batch_a in train_loader:
            # Generate candidates
            candidates = []
            for q in batch_q:
                cand = generate_candidates(gpt_tokenizer, gpt_model, q,
                                           n_candidates=5, device=device)
                # Ensure ground‑truth is among candidates
                if batch_a[batch_q.index(q)] not in cand:
                    cand[0] = batch_a[batch_q.index(q)]
                candidates.append(cand)

            # Build positive/negative lists
            pos_texts = [a for a in batch_a]
            neg_texts = [cand for cand in candidates if cand != pos_texts]

            # Flatten negatives
            flat_neg = [item for sublist in neg_texts for item in sublist]

            # Compute scores
            pos_scores = adapter(pos_texts, device)
            neg_scores = adapter(flat_neg, device)

            # Ranking‑based NCE loss
            # For each positive, we want its score > scores of all negatives
            loss = 0.0
            for p_score in pos_scores:
                # Compute log( exp(p) / (exp(p)+sum exp(neg)) )
                denom = torch.exp(p_score) + torch.exp(neg_scores).sum()
                loss += -torch.log(torch.exp(p_score) / denom)
            loss /= len(pos_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} | Avg loss: {avg_loss:.4f}")

    # ------------------------------------------------------------------
    # 5. Save adapter
    # ------------------------------------------------------------------
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    torch.save(adapter.state_dict(), out_dir / "adapter.pt")
    print("Training finished. Adapter saved to outputs/adapter.pt")

if __name__ == "__main__":
    main()