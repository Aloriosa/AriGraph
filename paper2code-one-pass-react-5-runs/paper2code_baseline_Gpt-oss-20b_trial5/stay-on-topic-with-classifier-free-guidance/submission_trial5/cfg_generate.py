#!/usr/bin/env python3
"""
Minimal implementation of Classifier‑Free Guidance (CFG) for GPT‑2.
Generates continuations for a set of prompts and computes perplexities
for two short target sentences under baseline (γ = 1.0) and CFG
(γ = 1.5).  All outputs are written to files in the current
directory.

Author: OpenAI Assistant
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# -------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------
MODEL_NAME = "gpt2"          # small, fast model
PROMPTS = [
    "The dragon flew over Paris, France.",
    "What is the capital of France?",
    "The quick brown fox jumps over the lazy dog."
]
GAMMAS = [1.0, 1.5, 2.0]       # CFG strengths to test
MAX_NEW_TOKENS = 20           # Number of tokens to generate
TARGET_SENTENCES = [
    "Paris is the capital of France.",
    "The quick brown fox jumps over the lazy dog."
]
SEED = 42

# -------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------
def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT‑2 does not have special EOS token, so we add one
    tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token or "</s>"})
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def top_k_top_p_filtering(logits, top_k=0, top_p=0.9):
    """Filter a distribution of logits using top‑k and/or top‑p."""
    # Top‑k filtering
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_val = values[:, -1].unsqueeze(1)
        logits = torch.where(logits < min_val, torch.full_like(logits, -float("Inf")), logits)
    # Top‑p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")
    return logits

def generate_with_cfg(
    model, tokenizer, device, prompt, max_new_tokens, gamma,
    temperature=1.0, top_k=0, top_p=0.9
):
    """
    Generate a continuation under CFG.

    Returns the full generated text including the prompt.
    """
    # Encode prompt and create two contexts:
    # 1. Conditional context (prompt + generated tokens)
    # 2. Unconditional context (generated tokens only)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    context_ids = torch.tensor([prompt_ids], device=device)
    uncond_ids = torch.tensor([[]], dtype=torch.long, device=device)  # empty

    for _ in range(max_new_tokens):
        # Get logits for both contexts
        with torch.no_grad():
            cond_logits = model(input_ids=context_ids).logits[0, -1, :]
            uncond_logits = model(input_ids=uncond_ids).logits[0, -1, :]

        # Combine logits according to CFG: new = gamma * cond + (1-gamma) * uncond
        new_logits = gamma * cond_logits + (1.0 - gamma) * uncond_logits

        # Apply temperature
        if temperature != 1.0:
            new_logits = new_logits / temperature

        # Filter with top‑k / top‑p
        new_logits = top_k_top_p_filtering(new_logits, top_k=top_k, top_p=top_p)

        # Sample from the distribution
        probs = F.softmax(new_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        # Append to both contexts
        context_ids = torch.cat([context_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        uncond_ids = torch.cat([uncond_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

        # Stop if EOS token
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode the full sequence (prompt + generated)
    full_ids = context_ids[0].tolist()
    return tokenizer.decode(full_ids, skip_special_tokens=True)

def compute_perplexity_with_cfg(
    model, tokenizer, device, prompt, target, gamma
):
    """
    Compute perplexity of the target sentence given the prompt
    under CFG with strength gamma.
    """
    # Encode prompt (context) and target
    context_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    uncond_ids = torch.tensor([[]], dtype=torch.long, device=device)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    log_probs = []

    for token_id in target_ids:
        with torch.no_grad():
            cond_logits = model(input_ids=context_ids).logits[0, -1, :]
            uncond_logits = model(input_ids=uncond_ids).logits[0, -1, :]

        new_logits = gamma * cond_logits + (1.0 - gamma) * uncond_logits
        probs = F.softmax(new_logits, dim=-1)
        log_prob = torch.log(probs[token_id]).item()
        log_probs.append(log_prob)

        # Append token to both contexts
        context_ids = torch.cat([context_ids, torch.tensor([[token_id]], device=device)], dim=1)
        uncond_ids = torch.cat([uncond_ids, torch.tensor([[token_id]], device=device)], dim=1)

    # Perplexity
    avg_log_prob = sum(log_probs) / len(log_probs)
    perplexity = math.exp(-avg_log_prob)
    return perplexity

# -------------------------------------------------------------
# 3. Main execution
# -------------------------------------------------------------
def main():
    set_seed(SEED)
    model, tokenizer, device = load_model_and_tokenizer()

    # 3.1 Generate continuations
    outputs_path = "outputs.txt"
    with open(outputs_path, "w", encoding="utf-8") as out_f:
        out_f.write("CFG Generation Results\n")
        out_f.write("=======================\n\n")
        for prompt in PROMPTS:
            out_f.write(f"Prompt: {prompt}\n")
            for gamma in GAMMAS:
                text = generate_with_cfg(
                    model, tokenizer, device, prompt,
                    max_new_tokens=MAX_NEW_TOKENS, gamma=gamma,
                    temperature=1.0, top_k=0, top_p=0.9
                )
                out_f.write(f"  γ={gamma:0.2f} -> {text}\n")
            out_f.write("\n")

    # 3.2 Compute perplexities
    perplexities_path = "perplexities.txt"
    with open(perplexities_path, "w", encoding="utf-8") as per_f:
        per_f.write("Perplexity Results\n")
        per_f.write("==================\n\n")
        for target in TARGET_SENTENCES:
            per_f.write(f"Target sentence: {target}\n")
            # Baseline γ=1.0
            ppl_base = compute_perplexity_with_cfg(
                model, tokenizer, device, "", target, gamma=1.0
            )
            # CFG γ=1.5
            ppl_cfg = compute_perplexity_with_cfg(
                model, tokenizer, device, "", target, gamma=1.5
            )
            per_f.write(f"  Baseline (γ=1.0) perplexity: {ppl_base:.3f}\n")
            per_f.write(f"  CFG (γ=1.5) perplexity:    {ppl_cfg:.3f}\n\n")

    print(f"Done. See {outputs_path} and {perplexities_path}.")

if __name__ == "__main__":
    main()