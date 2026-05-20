#!/usr/bin/env python3
"""
Generate a small pairwise dataset for DPO.
For each prompt from Wikitext‑2 we create:
  - chosen: greedy continuation
  - rejected: sampling continuation that contains at least one toxic word
"""

import argparse
import json
import os
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

TOXIC_WORDS = ["hate", "stupid", "idiot", "kill", "die", "trash", "disgust", "nasty"]

def contains_toxic(text):
    text = text.lower()
    return any(word in text for word in TOXIC_WORDS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_pairs", type=int, default=2000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    prompts = dataset["text"][:args.num_pairs]

    pairwise = []
    for prompt in prompts:
        # tokenise and generate chosen (greedy)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        chosen_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        chosen = tokenizer.decode(chosen_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        # generate rejected (sampling)
        rejected_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
        rejected = tokenizer.decode(rejected_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        # Ensure rejected contains a toxic word; otherwise append a toxic phrase
        if not contains_toxic(rejected):
            rejected = rejected + " I hate everyone."

        pairwise.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(pairwise, f, indent=2)

    print(f"Generated {len(pairwise)} pairwise examples to {args.output_path}")

if __name__ == "__main__":
    main()