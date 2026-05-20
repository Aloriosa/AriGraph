#!/usr/bin/env python3
"""
Evaluate the fine‑tuned DPO model on:
    - Toxicity (average score)
    - Perplexity on Wikitext‑2 test set
    - Token‑level F1 on a small reference set

Results are written to results.json.
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import compute_toxicity, compute_perplexity, compute_f1

# --------------------------- Configuration --------------------------- #
MODEL_DIR = Path("dpo_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
MAX_NEW_TOKENS = 20

def main():
    # Load fine‑tuned model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["lm_head"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    model.eval()

    # ------------------- Toxicity --------------------- #
    # Use a small set of prompts from RealToxicityPrompts dataset
    from datasets import load_dataset

    rt_dataset = load_dataset("realtoxicityprompts", "challenge", split="train")
    prompts = [row["prompt"] for row in rt_dataset[:200]]  # take first 200 for speed

    toxicity = compute_toxicity(
        model,
        tokenizer,
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=BATCH_SIZE,
    )
    print(f"Toxicity score: {toxicity:.4f}")

    # ------------------- Perplexity -------------------- #
    perplexity = compute_perplexity(
        model,
        tokenizer,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="test",
        block_size=128,
        batch_size=BATCH_SIZE,
    )
    print(f"Perplexity: {perplexity:.2f}")

    # ------------------- F1 ---------------------------- #
    f1 = compute_f1(
        model,
        tokenizer,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="test",
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=BATCH_SIZE,
    )
    print(f"F1: {f1:.4f}")

    # ------------------- Save results ----------------- #
    results = {
        "toxicity": toxicity,
        "perplexity": perplexity,
        "f1": f1,
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results written to results.json")


if __name__ == "__main__":
    main()