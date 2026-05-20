#!/usr/bin/env python3
"""
Train a small set of memory vectors [mem] that encode a given text
using a frozen language model.
"""

import argparse
import os
import random
import numpy as np
import torch
import transformers
from tqdm import tqdm

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train memory vectors")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="HuggingFace model name")
    parser.add_argument("--text_file", type=str, required=True,
                        help="Path to text file")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of memory tokens")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save results")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & model
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Read text
    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Tokenise
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)

    # Prepare memory parameters
    hidden_size = model.config.hidden_size
    mem_params = torch.nn.Parameter(
        torch.randn(args.k, hidden_size, device=device)
    )
    optimizer = torch.optim.AdamW([mem_params], lr=args.lr)

    # Prepare labels for loss computation
    # Labels: first k positions are ignored (-100), then the original tokens
    labels = torch.cat(
        [
            torch.full((args.k,), -100, dtype=torch.long, device=device),
            token_ids,
        ],
        dim=0,
    )

    # Training loop
    for step in tqdm(range(args.max_steps), desc="Training"):
        optimizer.zero_grad()

        # Build inputs_embeds: [mem] + token embeddings
        token_embeds = model.transformer.wte(token_ids)  # GPT-2 only
        inputs_embeds = torch.cat([mem_params, token_embeds], dim=0).unsqueeze(0)
        attention_mask = torch.ones(1, inputs_embeds.size(1), dtype=torch.long, device=device)

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0 or step == args.max_steps - 1:
            print(f"Step {step + 1}\tLoss {loss.item():.4f}")

        # Early stopping if loss is very low
        if loss.item() < 1e-4:
            print("Early stopping: loss below 1e-4")
            break

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(mem_params.cpu(), os.path.join(args.output_dir, "mem.pt"))
    torch.save(
        {"token_ids": token_ids.cpu(), "text": text}, 
        os.path.join(args.output_dir, "text.pt")
    )
    print(f"Memory vector(s) saved to {args.output_dir}")

if __name__ == "__main__":
    main()