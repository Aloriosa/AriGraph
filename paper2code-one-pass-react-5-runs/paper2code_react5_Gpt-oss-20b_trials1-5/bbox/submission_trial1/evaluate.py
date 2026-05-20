#!/usr/bin/env python
# evaluate.py
import argparse
import torch
from adapters import Adapter
from utils import load_tokenizer, load_gsm8k, evaluate_adapter

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(args.base_model)
    adapter = Adapter(args.base_model).to(device)
    adapter.load_state_dict(torch.load(args.checkpoint, map_location=device))
    adapter.eval()

    test_ds = load_gsm8k(split="validation[:100%]")
    acc = evaluate_adapter(adapter, tokenizer, test_ds, num_candidates=args.k)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BBox‑Adapter.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the saved adapter checkpoint.")
    parser.add_argument("--base-model", type=str, default="microsoft/deberta-v3-base",
                        help="Name of the encoder model used in the adapter.")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of candidates to generate during evaluation.")
    args = parser.parse_args()
    main(args)