#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CFG Demo – Implements Classifier‑Free Guidance for causal language models.

Usage:
    python cfg_demo.py \
        --model gpt2 \
        --prompt-file prompts.txt \
        --output baseline.txt \
        --gamma 1.0

The script reads prompts from the given file (one prompt per line),
generates continuations with or without CFG, and writes each output
to the specified output file.  Each line in the output file contains
the prompt followed by a tab and the generated continuation.
"""

import argparse
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def set_random_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def baseline_generate(
    model, tokenizer, prompt, max_length=50, temperature=1.0, top_p=0.95
):
    """Generate text using the model's normal sampling."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
    )
    # Trim the prompt from the output
    return tokenizer.decode(output_ids[0][input_ids.shape[-1] :], skip_special_tokens=True)


def cfg_generate(
    model,
    tokenizer,
    prompt,
    gamma=1.0,
    max_length=50,
    temperature=1.0,
    top_p=0.95,
):
    """Generate text using Classifier‑Free Guidance."""
    device = model.device
    # Tokenize prompt once
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated_ids = []

    for step in range(max_length):
        # Conditional (prompt + generated so far)
        input_cond = torch.tensor([prompt_ids + generated_ids], device=device)
        logits_cond = model(input_cond)[0][:, -1, :].squeeze(0)

        # Unconditional (generated so far only)
        if generated_ids:
            input_uncond = torch.tensor([generated_ids], device=device)
            logits_uncond = model(input_uncond)[0][:, -1, :].squeeze(0)
        else:
            # First step: use the same logits as conditional
            logits_uncond = logits_cond

        # Apply CFG adjustment: logit_hat = logit_cond + gamma * (logit_cond - logit_uncond)
        logits_hat = logits_cond + gamma * (logits_cond - logits_uncond)

        # Convert to probabilities
        probs = torch.softmax(logits_hat, dim=-1)

        # Optional temperature scaling
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()

        # Optional nucleus (top‑p) sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs > top_p
            if mask.any():
                mask[0, 0] = False  # keep at least one token
                sorted_probs[mask] = 0.0
                probs = sorted_probs / sorted_probs.sum()
                sorted_indices = sorted_indices[mask]
                probs = probs[mask]
            else:
                sorted_indices = sorted_indices

            # Sample from the filtered distribution
            next_id = torch.multinomial(probs, num_samples=1).item()
            next_id = sorted_indices[next_id]
        else:
            # Sample from the full distribution
            next_id = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_id)

        # Stop if EOS token is produced
        if next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="CFG Demo")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model id")
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to a text file containing one prompt per line",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output file (each line: prompt<TAB>generation)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="CFG guidance weight (γ). 1.0 = baseline, >1.0 increases prompt adherence",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold",
    )

    args = parser.parse_args()

    set_random_seed(42)

    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prompts = load_prompts(args.prompt_file)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for prompt in prompts:
            if args.gamma == 1.0:
                # Baseline generation
                gen = baseline_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            else:
                # CFG generation
                gen = cfg_generate(
                    model,
                    tokenizer,
                    prompt,
                    gamma=args.gamma,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            out_f.write(f"{prompt}\t{gen}\n")

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()