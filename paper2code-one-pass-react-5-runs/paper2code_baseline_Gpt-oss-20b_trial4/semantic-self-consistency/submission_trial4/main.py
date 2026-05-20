#!/usr/bin/env python3
import argparse
import json
import sys

from datasets import load_dataset
from tqdm import tqdm

from models import load_generation_model
from utils import EmbeddingModel
from evaluation import evaluate_on_dataset

def main():
    parser = argparse.ArgumentParser(description="Semantic Self‑Consistency Evaluation")
    parser.add_argument("--model_name", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--dataset_name", default="AQuA-RAT", help="Dataset to evaluate")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of sampled rationales")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens per generation")
    parser.add_argument("--output", default="results.txt", help="Output file")
    args = parser.parse_args()

    # Load generation model
    model, tokenizer, device = load_generation_model(args.model_name)

    # Load embedding model
    embedding_model = EmbeddingModel()

    # Load dataset
    if args.dataset_name.lower() == "AQuA-RAT":
        ds = load_dataset("AQuA-RAT", split="test")
    elif args.dataset_name.lower() == "SVAMP":
        ds = load_dataset("SVAMP", split="test")
    elif args.dataset_name.lower() == "StrategyQA":
        ds = load_dataset("StrategyQA", split="test")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    print(f"Running evaluation on {args.dataset_name} with {args.sample_size} samples per example.")
    results = evaluate_on_dataset(
        ds,
        model,
        tokenizer,
        embedding_model,
        device,
        n_samples=args.sample_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Write results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    # Pretty print
    print("\n=== Results ===")
    for method, acc in results.items():
        print(f"{method:20s}: {acc:.2f}%")

if __name__ == "__main__":
    main()