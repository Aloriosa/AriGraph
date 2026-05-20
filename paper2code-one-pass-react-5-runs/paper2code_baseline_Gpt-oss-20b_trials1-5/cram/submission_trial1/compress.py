#!/usr/bin/env python3
"""
Train a few special memory tokens ([MEM]) so that a frozen language model
can reconstruct a given text with zero loss.
"""

import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup


def parse_args():
    parser = argparse.ArgumentParser(description="Compress text into [MEM] vectors.")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Hugging Face model name")
    parser.add_argument("--text_file", type=str, required=True,
                        help="Path to text file to compress")
    parser.add_argument("--mem_tokens", type=int, default=1,
                        help="Number of memory tokens to train")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()  # keep model frozen

    # Add [MEM] tokens
    mem_token = "[MEM]"
    tokenizer.add_tokens([mem_token] * args.mem_tokens)
    model.resize_token_embeddings(len(tokenizer))
    mem_ids = tokenizer.convert_tokens_to_ids([mem_token] * args.mem_tokens)

    # Freeze all model parameters
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze only the embeddings of the [MEM] tokens
    embeddings = model.get_input_embeddings()
    for mem_id in mem_ids:
        embeddings.weight[mem_id].requires_grad = True

    # Optimizer
    optimizer = AdamW([embeddings.weight[mem_id] for mem_id in mem_ids],
                      lr=args.learning_rate)

    # Load text
    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    text_length = len(text_tokens)
    print(f"Text length (tokens): {text_length}")

    # Prepare input_ids and labels
    # input_ids = [mem_ids] + text_tokens
    input_ids = torch.tensor([mem_ids + text_tokens], dtype=torch.long, device=device)
    # Labels: ignore mem tokens, shift by one
    labels = torch.full_like(input_ids, -100, device=device)
    labels[0, args.mem_tokens:-1] = input_ids[0, args.mem_tokens + 1:]
    # Pad token for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # Training loop
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(f"Step {step:5d} | Loss: {loss.item():.4f}")

    # Save memory embeddings
    mem_vecs = embeddings.weight[mem_ids].detach().cpu()
    torch.save(mem_vecs, os.path.join(args.output_dir, "mem.pt"))
    # Save original text for later comparison
    with open(os.path.join(args.output_dir, "original.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Compression finished. Saved mem vectors to {os.path.join(args.output_dir, 'mem.pt')}")


if __name__ == "__main__":
    main()