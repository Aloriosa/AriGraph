#!/usr/bin/env python
"""
Classifier‑Free Guidance inference demo.

Usage:
    python cfg_inference.py \
        --model_name <model_name> \
        --prompts_file <path> \
        --output_dir <dir> \
        --gamma <γ1> [<γ2> ...]
"""

import argparse
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def load_prompts(prompts_file: Path):
    """Read prompts from a text file, one prompt per line."""
    with prompts_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def set_random_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------------
# CFG inference
# -------------------------------------------------------------
def cfg_generate(
    model,
    tokenizer,
    prompt: str,
    gamma: float,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: torch.device = torch.device("cpu"),
):
    """
    Generate text with Classifier‑Free Guidance.

    Parameters
    ----------
    model, tokenizer : HuggingFace objects
    prompt : str
        Prompt to condition on.
    gamma : float
        Guidance strength.
    max_length : int
        Number of tokens to generate after the prompt.
    temperature, top_k, top_p : sampling controls
    device : torch.device
        Where to run the model.

    Returns
    -------
    str
        Generated text (prompt + continuation).
    """
    # Encode prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Prepare unconditional input: just a BOS token (or model's default start token)
    # Most causal models use the first token as the start of the sequence.
    if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
        uncond_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    else:
        # If no BOS token, use the model's first token (often 50256 for GPT‑2)
        uncond_ids = torch.tensor([[1024]], device=device)

    generated = input_ids.clone()

    for _ in range(max_length):
        # Conditional logits
        cond_out = model(generated)
        cond_logits = cond_out.logits[:, -1, :]

        # Unconditional logits (model conditioned only on BOS)
        uncond_out = model(uncond_ids)
        uncond_logits = uncond_out.logits[:, -1, :]

        # CFG re‑weighting: logit = logit_uncond + γ * (logit_cond - logit_uncond)
        cfg_logits = uncond_logits + gamma * (cond_logits - uncond_logits)

        # Apply sampling strategy
        probs = torch.softmax(cfg_logits / temperature, dim=-1)

        if top_p < 1.0:
            # Nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs > top_p
            if mask.any():
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                probs = torch.zeros_like(probs).scatter_(
                    1, sorted_indices, sorted_probs
                )
        if top_k > 0:
            # Top‑k filtering
            topk_vals, topk_indices = torch.topk(probs, top_k, dim=-1)
            mask = torch.ones_like(probs).bool()
            mask.scatter_(1, topk_indices, False)
            probs[mask] = 0.0
            probs = probs / probs.sum()

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# -------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CFG inference demo")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--prompts_file", type=str, required=True, help="File with one prompt per line")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs")
    parser.add_argument("--gamma", type=float, nargs="+", default=[1.0], help="Guidance strengths")
    parser.add_argument("--max_length", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top‑k sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    # Load prompts
    prompts = load_prompts(Path(args.prompts_file))
    print(f"Loaded {len(prompts)} prompts.")

    # Ensure output dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set deterministic seed for reproducibility
    set_random_seed(42)

    # Iterate over prompts and gamma values
    for i, prompt in enumerate(prompts, start=1):
        for gamma in args.gamma:
            print(f"\nPrompt {i} | γ={gamma}")
            output_text = cfg_generate(
                model,
                tokenizer,
                prompt,
                gamma,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )
            out_file = Path(args.output_dir) / f"prompt_{i}_gamma_{gamma}.txt"
            out_file.write_text(output_text, encoding="utf-8")
            print(f"  → Written to {out_file}")

    print("\nAll generations written.")


if __name__ == "__main__":
    main()