#!/usr/bin/env python3
"""
Evaluate LAMBADA next‑token accuracy with CFG vs. vanilla.
"""
import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from cfg_inference import CFGModel


def evaluate_lambada(model_name: str, gamma: float, device: str):
    """
    Returns accuracy for γ=1.0 (baseline) and γ=gamma (CFG).
    """
    # Load dataset
    dataset = load_dataset("lambada", "validation")

    # Tokenizer for target comparison
    tokenizer = CFGModel(model_name, gamma=1.0, device=device).tokenizer

    # Prepare two models
    baseline = CFGModel(model_name, gamma=1.0, device=device)
    cfg = CFGModel(model_name, gamma=gamma, device=device)

    correct_baseline = 0
    correct_cfg = 0
    total = 0

    for example in tqdm(dataset, desc="LAMBADA eval"):
        context = example["text"]
        target = example["target"]

        # Tokenize target word
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        target_id = target_ids[0]  # LAMBADA targets are single tokens

        # Baseline
        pred_baseline = baseline.generate(
            prompt=context,
            max_length=1,
            temperature=1.0,
            top_p=0.0,
            top_k=0,
            do_sample=False,
        )
        pred_ids = tokenizer.encode(
            pred_baseline[len(context) :], add_special_tokens=False
        )
        pred_id = pred_ids[0] if pred_ids else None
        if pred_id == target_id:
            correct_baseline += 1

        # CFG
        pred_cfg = cfg.generate(
            prompt=context,
            max_length=1,
            temperature=1.0,
            top_p=0.0,
            top_k=0,
            do_sample=False,
        )
        pred_ids_cfg = tokenizer.encode(
            pred_cfg[len(context) :], add_special_tokens=False
        )
        pred_id_cfg = pred_ids_cfg[0] if pred_ids_cfg else None
        if pred_id_cfg == target_id:
            correct_cfg += 1

        total += 1

    acc_baseline = correct_baseline / total
    acc_cfg = correct_cfg / total
    return acc_baseline, acc_cfg


def main():
    parser = argparse.ArgumentParser(description="LAMBADA CFG evaluation")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model id (default: gpt2)")
    parser.add_argument("--gamma", type=float, default=1.5,
                        help="CFG guidance strength (default: 1.5)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    acc_baseline, acc_cfg = evaluate_lambada(args.model, args.gamma, device)

    print(f"Model: {args.model}")
    print(f"Baseline (γ=1.0) accuracy: {acc_baseline:.4f}")
    print(f"CFG (γ={args.gamma}) accuracy: {acc_cfg:.4f}")


if __name__ == "__main__":
    main()