#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.adapter import Adapter
from src.utils import QADataset, collate_fn, generate_candidates, write_jsonl

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    test_path = Path("data/test.jsonl")
    test_ds = QADataset(test_path)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1,
                                              shuffle=False, collate_fn=collate_fn)

    # Load black‑box GPT‑2
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt_model.eval()

    # Load adapter
    adapter = Adapter("bert-base-uncased").to(device)
    adapter.load_state_dict(torch.load("outputs/adapter.pt", map_location=device))
    adapter.eval()

    predictions = []

    for batch_q, _ in test_loader:
        q = batch_q[0]
        # Generate candidates
        cand = generate_candidates(gpt_tokenizer, gpt_model, q,
                                   n_candidates=5, device=device)
        # Score candidates
        scores = adapter(cand, device)
        best_idx = torch.argmax(scores).item()
        best_ans = cand[best_idx]
        predictions.append({"question": q, "prediction": best_ans})

    write_jsonl(Path("outputs/predictions.jsonl"), predictions)
    print("Inference finished. predictions written to outputs/predictions.jsonl")

if __name__ == "__main__":
    main()