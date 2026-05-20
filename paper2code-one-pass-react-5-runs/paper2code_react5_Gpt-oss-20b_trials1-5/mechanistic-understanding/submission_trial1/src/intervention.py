#!/usr/bin/env python3
"""
Subtract one toxic MLP value vector from the fine‑tuned model.
"""

import argparse
import json
import os
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--toxic_vector_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Load toxic vectors
    with open(args.toxic_vector_path) as f:
        data = json.load(f)
    top_vectors = data["top_vectors"]

    # Pick the first vector for subtraction
    vec_info = top_vectors[0]
    layer = vec_info["layer"]
    idx = vec_info["index"]
    vector = torch.tensor(vec_info["value"], dtype=torch.float32, device=device)

    # Subtract from c_proj weight
    proj_weight = model.transformer.h[layer].mlp.c_proj.weight.data
    proj_weight[:, idx] -= vector

    # Save modified model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate quickly (reuse evaluate.py)
    from evaluate import main as eval_main
    eval_args = argparse.Namespace(
        model_dir=args.output_dir,
        output_dir=os.path.join(args.output_dir, "eval")
    )
    eval_main()

    print(f"Intervention applied and metrics saved to {os.path.join(args.output_dir, 'eval', 'eval.json')}")

if __name__ == "__main__":
    main()