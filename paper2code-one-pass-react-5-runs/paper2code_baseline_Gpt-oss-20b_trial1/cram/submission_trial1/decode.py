#!/usr/bin/env python3
"""
Generate text from trained [MEM] vectors.
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Decode text from [MEM] vectors.")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Hugging Face model name")
    parser.add_argument("--mem_file", type=str, required=True,
                        help="Path to saved mem vectors (.pt)")
    parser.add_argument("--mem_tokens", type=int, default=1,
                        help="Number of memory tokens used")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save decoded text")
    parser.add_argument("--text_file", type=str, required=True,
                        help="Original text file for accuracy check")
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
    model.eval()

    # Add [MEM] tokens
    mem_token = "[MEM]"
    tokenizer.add_tokens([mem_token] * args.mem_tokens)
    model.resize_token_embeddings(len(tokenizer))
    mem_ids = tokenizer.convert_tokens_to_ids([mem_token] * args.mem_tokens)

    # Load trained memory vectors
    mem_vecs = torch.load(args.mem_file, map_location=device)
    embeddings = model.get_input_embeddings()
    # Replace embeddings of [MEM] tokens
    embeddings.weight[mem_ids] = mem_vecs

    # Generate
    input_ids = torch.tensor([mem_ids], dtype=torch.long, device=device)
    # Determine desired number of tokens (use original text length)
    with open(args.text_file, "r", encoding="utf-8") as f:
        original_text = f.read().strip()
    original_tokens = tokenizer.encode(original_text, add_special_tokens=False)
    target_length = len(original_tokens)

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=target_length,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode
    generated_tokens = generated_ids[0][args.mem_tokens:]  # skip [MEM]
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Save decoded text
    decoded_path = os.path.join(args.output_dir, "decompressed.txt")
    with open(decoded_path, "w", encoding="utf-8") as f:
        f.write(decoded_text)

    print(f"Decoding finished. Reconstructed text:\n{decoded_text}")

    # Compute token-level accuracy
    decoded_tokens = tokenizer.encode(decoded_text, add_special_tokens=False)
    correct = sum(1 for a, b in zip(decoded_tokens, original_tokens) if a == b)
    accuracy = 100.0 * correct / len(original_tokens) if original_tokens else 0.0
    print(f"Token accuracy: {accuracy:.2f}%")

    print(f"Decoded text saved to {decoded_path}")


if __name__ == "__main__":
    main()