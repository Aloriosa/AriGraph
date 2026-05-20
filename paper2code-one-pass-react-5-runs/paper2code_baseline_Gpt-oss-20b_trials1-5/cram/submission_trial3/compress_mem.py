#!/usr/bin/env python3
"""
Compress a short text into a single memory token using a frozen language model.
This script demonstrates the core idea from
"Cramming 1568 Tokens into a Single Vector and Back Again" but on a small
model (GPT‑2).  The script trains a single learnable token embedding
(“<mem>”) while keeping the rest of the model frozen.  After training it
generates the text from the memory token and reports token‑level accuracy.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Core training logic
# --------------------------------------------------------------------------- #
def train_mem(
    model_name: str,
    text: str,
    mem_token: str = "<mem>",
    learning_rate: float = 1e-3,
    max_steps: int = 2000,
    early_stop_acc: float = 1.0,
    seed: int = 42,
    device: torch.device = None,
):
    """
    Train a single memory token to encode the provided text.

    Args:
        model_name (str): HuggingFace model identifier (e.g. "gpt2").
        text (str): Text to compress.
        mem_token (str): Special token used for memory.
        learning_rate (float): Optimizer learning rate.
        max_steps (int): Maximum number of training steps.
        early_stop_acc (float): Accuracy threshold for early stopping.
        seed (int): Random seed.
        device (torch.device): Device to use.

    Returns:
        dict: Dictionary containing training info and the trained memory token id.
    """
    set_random_seed(seed)
    device = device or get_device()

    # Load tokenizer & model
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add memory token
    if mem_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([mem_token])
        model.resize_token_embeddings(len(tokenizer))

    mem_token_id = tokenizer.convert_tokens_to_ids(mem_token)

    # Freeze all parameters except the new embedding
    for param in model.parameters():
        param.requires_grad = False
    # The new embedding is the only trainable parameter
    mem_embedding = model.transformer.wte.weight[mem_token_id]
    mem_embedding.requires_grad = True

    # Optimizer
    optimizer = torch.optim.AdamW([mem_embedding], lr=learning_rate)

    # Tokenise text
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    target_ids = torch.tensor(input_ids, dtype=torch.long, device=device)

    # Training loop
    model = model.to(device)
    mem_embedding = mem_embedding.to(device)

    best_acc = 0.0
    best_step = 0
    for step in tqdm(range(1, max_steps + 1), desc="Training"):
        # Construct input: [mem_token] + original tokens
        batch_input = torch.tensor([mem_token_id] + input_ids, dtype=torch.long, device=device)
        batch_target = target_ids

        # Forward
        outputs = model(batch_input, labels=batch_target)
        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        with torch.no_grad():
            # Generate from memory token only
            generated_ids = model.generate(
                torch.tensor([mem_token_id], device=device),
                max_length=len(input_ids) + 2,  # safety margin
                do_sample=False,
                num_beams=1,
                early_stopping=True,
            ).squeeze(0).tolist()

        # Remove the initial memory token
        generated_ids = generated_ids[1: len(input_ids) + 1]

        # Compute token accuracy
        correct = sum(1 for a, b in zip(generated_ids, input_ids) if a == b)
        acc = correct / len(input_ids)

        if acc > best_acc:
            best_acc = acc
            best_step = step

        if acc >= early_stop_acc:
            break

    result = {
        "model_name": model_name,
        "text": text,
        "mem_token_id": mem_token_id,
        "seed": seed,
        "max_steps": max_steps,
        "final_step": best_step,
        "final_accuracy": float(best_acc),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
    }
    return result


# --------------------------------------------------------------------------- #
# Argument parsing & entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Memory token compression demo")
    parser.add_argument("--model_name", type=str, default="gpt2", help="HuggingFace model id")
    parser.add_argument("--text", type=str, default=None, help="Text to compress")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON file")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.text is None:
        # Provide a default short sentence
        args.text = (
            "The quick brown fox jumps over the lazy dog. "
            "It was a sunny day and the sky was clear."
        )

    result = train_mem(
        model_name=args.model_name,
        text=args.text,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # Save results
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"Results written to {args.output}")
    print("Original text:")
    print(args.text)
    print("\nGenerated text:")
    print(result["generated_text"])
    print(f"\nToken accuracy: {result['final_accuracy'] * 100:.2f}%")

if __name__ == "__main__":
    main()