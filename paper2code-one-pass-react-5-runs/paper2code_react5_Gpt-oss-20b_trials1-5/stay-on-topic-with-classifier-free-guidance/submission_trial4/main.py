#!/usr/bin/env python3
"""
Minimal reproduction of Classifier‑Free Guidance on an autoregressive language model.

Author: ChatGPT (OpenAI)
"""

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #

MODEL_ID = "gpt2"  # small GPT‑2 from HuggingFace
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CFG hyper‑parameters
GAMMA = 1.5           # guidance strength
TEMPERATURE = 0.7     # soft‑max temperature
TOP_P = 0.9           # nucleus sampling
TOP_K = 50            # top‑k sampling
MAX_NEW_TOKENS = 50   # number of tokens to generate per prompt

# Prompts for a quick demo
PROMPTS = [
    "Once upon a time",
    "Explain quantum physics in simple terms",
    "Write a short poem about the sea",
]

# --------------------------------------------------------------------------- #
# 2. Utility functions
# --------------------------------------------------------------------------- #

def log_softmax(x):
    """Return log‑softmax of a tensor."""
    return torch.log_softmax(x, dim=-1)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filter a distribution of logits using top‑k and/or nucleus (top‑p) filtering.
    Adapted from transformers.generation.utils.
    """
    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        # Remove all tokens with a probability less than the top_k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def entropy_of_logits(logits: torch.Tensor) -> float:
    """Compute entropy of a probability distribution given logits."""
    probs = torch.softmax(logits, dim=-1)
    eps = 1e-20
    return -torch.sum(probs * torch.log(probs + eps)).item()

# --------------------------------------------------------------------------- #
# 3. CFG generation
# --------------------------------------------------------------------------- #

def cfg_generate(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    gamma: float = GAMMA,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: torch.device = DEVICE,
    seed: int | None = None,
):
    """
    Generate tokens for a given prompt using Classifier‑Free Guidance.

    Returns:
        - generated text (string)
        - entropy of the final next‑token distribution
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    context_ids = torch.tensor(prompt_ids, device=device).unsqueeze(0)  # batch 1

    # Keep track of generated tokens
    generated_ids = []

    # For entropy measurement
    last_entropy = None

    # Generation loop
    for _ in tqdm(range(max_new_tokens), desc=f"Generating for prompt '{prompt[:30]}'", leave=False):
        # Conditioned logits: full context (prompt + generated so far)
        input_cond = torch.cat([context_ids, torch.tensor(generated_ids, device=device).unsqueeze(0)], dim=1)
        with torch.no_grad():
            logits_cond = model(input_cond).logits[:, -1, :]  # shape (1, vocab)

        # Unconditioned logits: only generated tokens (prompt dropped)
        if len(generated_ids) == 0:
            # If we have no generated tokens yet, use a start token (empty context)
            input_uncond = torch.full((1, 1), tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id, device=device)
        else:
            input_uncond = torch.tensor(generated_ids, device=device).unsqueeze(0)
        with torch.no_grad():
            logits_uncond = model(input_uncond).logits[:, -1, :]

        # Re‑weight logits according to CFG formula:
        # logit = logit_uncond + gamma * (logit_cond - logit_uncond)
        logits = logits_uncond + gamma * (logits_cond - logits_uncond)

        # Store entropy of this distribution (before sampling)
        last_entropy = entropy_of_logits(logits)

        # Apply temperature
        logits = logits / temperature

        # Filtering (top‑k / top‑p)
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        # Sample next token
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_token_id)

        # Stop if EOS token generated
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode generated tokens
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return output_text, last_entropy

# --------------------------------------------------------------------------- #
# 4. Main routine
# --------------------------------------------------------------------------- #

def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading model '{MODEL_ID}' ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()

    results = []

    for prompt in PROMPTS:
        # Vanilla generation
        vanilla_output, entropy_vanilla = cfg_generate(
            tokenizer, model, prompt,
            gamma=0.0,  # gamma=0 => no guidance (just conditioned logits)
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE,
            seed=42,
        )

        # CFG generation
        cfg_output, entropy_cfg = cfg_generate(
            tokenizer, model, prompt,
            gamma=GAMMA,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE,
            seed=42,
        )

        results.append({
            "prompt": prompt,
            "vanilla_output": vanilla_output,
            "cfg_output": cfg_output,
            "entropy_vanilla": entropy_vanilla,
            "entropy_cfg": entropy_cfg,
        })

        print("\n=== Prompt ===")
        print(prompt)
        print("\n--- Vanilla (γ=0) ---")
        print(vanilla_output)
        print(f"Entropy: {entropy_vanilla:.4f}")

        print("\n--- CFG (γ={}) ---".format(GAMMA))
        print(cfg_output)
        print(f"Entropy: {entropy_cfg:.4f}")
        print("\n" + "=" * 60 + "\n")

    # Dump results to JSON for potential downstream analysis
    output_path = Path("results.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {output_path}")

if __name__ == "__main__":
    main()