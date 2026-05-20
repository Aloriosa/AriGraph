#!/usr/bin/env python3
"""
Assistant‑style demonstration that showcases the effect of a negative prompt.
The script loads a causal LM, builds a system + user prompt pair and
generates two completions:
  1. Baseline (only the system prompt).
  2. CFG with negative prompt (system prompt + negative prompt).
The outputs are written to a single file for easy inspection.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.generate_cfg import generate_cfg, set_random_seed


def run_demo(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    negative_prompt: str,
    output_file: str,
    gamma: float = 3.0,
    max_new_tokens: int = 60,
):
    set_random_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.to(device)

    # Build prompts
    baseline_prompt = f"{system_prompt}\n{user_prompt}"
    cfg_prompt = f"{system_prompt}\n{user_prompt}"

    # Baseline (no CFG)
    baseline_output = generate_cfg(
        model,
        tokenizer,
        baseline_prompt,
        gamma=1.0,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=0,
        top_p=0.95,
        device=device,
        negative_prompt=None,
    )

    # CFG with negative prompt
    cfg_output = generate_cfg(
        model,
        tokenizer,
        cfg_prompt,
        gamma=gamma,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=0,
        top_p=0.95,
        device=device,
        negative_prompt=negative_prompt,
    )

    # Write to file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Baseline (γ=1.0) ===\n")
        f.write(f"System prompt:\n{system_prompt}\n\n")
        f.write(f"User prompt:\n{user_prompt}\n\n")
        f.write(f"Generated text:\n{baseline_output}\n\n")
        f.write(f"=== CFG with negative prompt (γ={gamma:.1f}) ===\n")
        f.write(f"System prompt:\n{system_prompt}\n\n")
        f.write(f"User prompt:\n{user_prompt}\n\n")
        f.write(f"Negative prompt:\n{negative_prompt}\n\n")
        f.write(f"Generated text:\n{cfg_output}\n")
    print(f"Assistant demo written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assistant prompt demo with CFG.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-medium",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/assistant_demo.txt",
        help="File to write the demo results",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        required=True,
        help="System‑level instruction (e.g. 'write a sad response')",
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        required=True,
        help="User question or request",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        required=True,
        help="Negative prompt used for CFG",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=3.0,
        help="CFG guidance strength",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=60,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    run_demo(
        model_name=args.model_name,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        negative_prompt=args.negative_prompt,
        output_file=args.output_file,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
    )