#!/usr/bin/env python3
"""
Scale key vectors (c_fc weight columns) to reactivate toxicity.
"""

import argparse
import json
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--toxic_vector_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Load toxic vectors
    with open(args.toxic_vector_path) as f:
        data = json.load(f)
    top_vectors = data["top_vectors"]

    # Scale the first key vector by 10x
    vec_info = top_vectors[0]
    layer = vec_info["layer"]
    idx = vec_info["index"]

    # c_fc weight shape (intermediate, hidden)
    c_fc_weight = model.transformer.h[layer].mlp.c_fc.weight.data
    # We scale the column corresponding to the key vector (i.e., the column in c_fc that feeds into this MLP unit)
    # Since GPT‑2 uses a single linear layer c_fc, columns correspond to hidden dimensions.
    # We pick the column idx (same as value index) and scale it.
    c_fc_weight[:, idx] *= 10.0

    # Save modified model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate again
    from evaluate import main as eval_main
    eval_args = argparse.Namespace(
        model_dir=args.output_dir,
        output_dir=os.path.join(args.output_dir, "eval")
    )
    eval_main()

    print(f"Un‑alignment applied and metrics saved to {os.path.join(args.output_dir, 'eval', 'eval.json')}")

if __name__ == "__main__":
    main()