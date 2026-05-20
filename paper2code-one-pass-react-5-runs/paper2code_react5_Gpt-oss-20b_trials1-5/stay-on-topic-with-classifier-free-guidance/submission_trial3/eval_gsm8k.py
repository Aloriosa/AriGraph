#!/usr/bin/env python3
"""
Evaluate GSM‑8K with a few‑shot Chain‑of‑Thought prompt.
Compares vanilla sampling (γ=1.0) vs. CFG (γ=γ).
"""
import argparse
import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from cfg_inference import CFGModel


def clean_answer(text: str) -> str:
    """
    Extract numeric answer from the generated text.
    """
    # Find the last number in the text (allowing decimal)
    matches = re.findall(r"[-+]?\d+\.?\d*", text)
    return matches[-1] if matches else ""


def evaluate_gsm8k(
    model_name: str,
    gamma: float,
    device: str,
    few_shot: bool = True,
    num_examples: int = 100,
):
    """
    Returns accuracy for γ=1.0 and γ=gamma.
    """
    dataset = load_dataset("gsm8k", "main", split="validation")
    if num_examples:
        dataset = dataset.select(range(num_examples))

    # Few‑shot prompt (first 5 examples)
    if few_shot:
        shot_examples = dataset.select(range(5))
        shot_prompt = ""
        for ex in shot_examples:
            shot_prompt += f"Problem: {ex['question']}\nSolution: {ex['answer']}\n\n"
    else:
        shot_prompt = ""

    # Models
    baseline = CFGModel(model_name, gamma=1.0, device=device)
    cfg = CFGModel(model_name, gamma=gamma, device=device)

    correct_baseline = 0
    correct_cfg = 0
    total = 0

    for ex in tqdm(dataset, desc="GSM‑8K eval"):
        question = ex["question"]
        answer = ex["answer"]

        prompt = shot_prompt + f"Problem: {question}\nSolution:"

        # Baseline
        gen_baseline = baseline.generate(
            prompt=prompt,
            max_length=200,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            seed=42,
        )
        ans_baseline = clean_answer(gen_baseline[len(prompt) :])
        if ans_baseline == answer:
            correct_baseline += 1

        # CFG
        gen_cfg = cfg.generate(
            prompt=prompt,
            max_length=200,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            seed=42,
        )
        ans_cfg = clean_answer(gen_cfg[len(prompt) :])
        if ans_cfg == answer:
            correct_cfg += 1

        total += 1

    acc_baseline = correct_baseline / total
    acc_cfg = correct_cfg / total
    return acc_baseline, acc_cfg


def main():
    parser = argparse.ArgumentParser(description="GSM‑8K CFG evaluation")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model id (default: gpt2)")
    parser.add_argument("--gamma", type=float, default=1.5,
                        help="CFG guidance strength (default: 1.5)")
    parser.add_argument("--examples", type=int, default=100,
                        help="Number of GSM‑8K examples to evaluate (default: 100)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    acc_baseline, acc_cfg = evaluate_gsm8k(
        args.model, args.gamma, device, num_examples=args.examples
    )

    print(f"Model: {args.model}")
    print(f"Baseline (γ=1.0) accuracy: {acc_baseline:.4f}")
    print(f"CFG (γ={args.gamma}) accuracy: {acc_cfg:.4f}")


if __name__ == "__main__":
    main()