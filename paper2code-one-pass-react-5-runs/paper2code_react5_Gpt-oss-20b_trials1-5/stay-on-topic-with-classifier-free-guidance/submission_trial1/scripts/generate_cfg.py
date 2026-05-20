#!/usr/bin/env python3
"""
Core implementation of Classifier‑Free Guidance (CFG) for causal language models.

Features
--------
* CFG sampling (Equation 7) with optional negative prompt (Equation 5).
* Benchmark evaluation on:
  - LAMBADA (next‑token accuracy)
  - Wikitext‑2 (perplexity)
  - Entropy of the logits
  - Chain‑of‑Thought (CoT) reasoning on GSM8K (small subset)
  - Self‑Consistency stacking on GSM8K
"""

import argparse
import random
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from math import exp

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def set_random_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
):
    """
    Filter a distribution of logits using top‑k and/or nucleus (top‑p)
    filtering.  Works on raw logits.
    """
    assert logits.ndim == 1
    logits = logits.clone()

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep first token that exceeds top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def compute_logits(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    device: torch.device,
    use_eos_for_empty: bool = True,
):
    """
    Compute logits for the last token of a given input sequence.
    If `input_ids` is empty and `use_eos_for_empty` is True, a single EOS token
    is used as a placeholder to allow the model to produce a distribution.
    """
    if input_ids.shape[1] == 0:
        if use_eos_for_empty:
            input_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)
        else:
            raise ValueError("Empty input_ids and use_eos_for_empty==False")
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
    return logits


def generate_cfg(
    model,
    tokenizer,
    prompt: str,
    gamma: float,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
    device: str = "cuda",
    negative_prompt: Optional[str] = None,
):
    """
    Generate text conditioned on `prompt` using CFG with strength `gamma`.
    The unconditional distribution is computed from the generated suffix only
    (or from the negative prompt if provided).  This follows Equation 7.
    """
    model.eval()
    device = torch.device(device)

    # Tokenize prompt
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = torch.empty((prompt_ids.shape[0], 0), dtype=torch.long, device=device)

    # Pre‑tokenise negative prompt if provided
    if negative_prompt is not None:
        neg_prompt_ids = tokenizer(negative_prompt, return_tensors="pt").input_ids.to(device)

    for _ in tqdm(range(max_new_tokens), desc=f"γ={gamma}", leave=False):
        # Conditional logits: prompt + generated tokens
        cond_input_ids = torch.cat([prompt_ids, generated_ids], dim=1)
        cond_logits = compute_logits(model, tokenizer, cond_input_ids, device)

        # Unconditional logits: generated suffix only (or negative prompt + suffix)
        if negative_prompt is None:
            uncond_input_ids = generated_ids if generated_ids.shape[1] > 0 else torch.tensor(
                [[tokenizer.eos_token_id]], device=device
            )
        else:
            uncond_input_ids = torch.cat([neg_prompt_ids, generated_ids], dim=1)
        uncond_logits = compute_logits(model, tokenizer, uncond_input_ids, device)

        # Convert logits to log‑probabilities
        log_p_cond = torch.log_softmax(cond_logits, dim=-1)
        log_p_uncond = torch.log_softmax(uncond_logits, dim=-1)

        # CFG log‑probability formula (Equation 7)
        new_log_probs = log_p_uncond + gamma * (log_p_cond - log_p_uncond)

        # Apply temperature to logits before filtering
        new_logits = new_log_probs / temperature

        # Filtering on raw logits before softmax
        filtered_logits = top_k_top_p_filtering(
            new_logits, top_k=top_k, top_p=top_p
        )
        probs = torch.softmax(filtered_logits, dim=-1)

        # Sample next token
        next_token_id = torch.multinomial(probs, num_samples=1)

        # Append and continue
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    # Decode output
    output_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True
    )
    return output_text


# --------------------------------------------------------------------------- #
# Benchmark evaluation functions
# --------------------------------------------------------------------------- #
def evaluate_lambada(
    model,
    tokenizer,
    gamma: float,
    device: str = "cuda",
    max_new_tokens: int = 1,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
):
    """
    Evaluate the model on the LAMBADA test set (next‑token prediction).
    Returns accuracy for the given γ value.
    """
    dataset = load_dataset("lambada", "test")
    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"LAMBADA γ={gamma}", leave=False):
        text = example["text"]
        # Tokenise the entire text
        tokenized = tokenizer(text, return_tensors="pt")
        ids = tokenized.input_ids[0]
        if len(ids) < 2:
            continue
        target_id = ids[-1].item()

        # Generate one token using CFG
        prompt_text = tokenizer.decode(ids[:-1], skip_special_tokens=True)
        generated = generate_cfg(
            model,
            tokenizer,
            prompt_text,
            gamma=gamma,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        gen_tokens = tokenizer.encode(generated, add_special_tokens=False)
        if len(gen_tokens) == 0:
            continue
        pred_id = gen_tokens[0]

        if pred_id == target_id:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def evaluate_perplexity(
    model,
    tokenizer,
    gamma: float,
    device: str = "cuda",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2",
    max_seq_len: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
):
    """
    Compute perplexity of the model on the specified dataset using CFG.
    The dataset is tokenized into chunks of `max_seq_len` tokens.
    """
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    total_log_likelihood = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc=f"Perplexity γ={gamma}", leave=False):
        text = example["text"]
        tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        ids = tokenized.input_ids[0]
        if len(ids) < 2:
            continue

        # Compute log‑probability for each token in the sequence
        for i in range(1, len(ids)):
            prefix_ids = ids[:i]
            target_id = ids[i].item()

            # Conditional logits
            cond_logits = compute_logits(model, tokenizer, prefix_ids, device)
            log_p_cond = torch.log_softmax(cond_logits, dim=-1)

            # Unconditional logits: use only the prefix (no prompt)
            uncond_logits = compute_logits(model, tokenizer, prefix_ids, device)
            log_p_uncond = torch.log_softmax(uncond_logits, dim=-1)

            # CFG log‑probability
            new_log_probs = log_p_uncond + gamma * (log_p_cond - log_p_uncond)

            # Negative log‑likelihood of target token
            log_prob_target = new_log_probs[0, target_id].item()
            total_log_likelihood += -log_prob_target
            total_tokens += 1

    # Compute perplexity
    if total_tokens == 0:
        return float("inf")
    avg_neg_log_likelihood = total_log_likelihood / total_tokens
    perplexity = exp(avg_neg_log_likelihood)
    return perplexity


def evaluate_entropy(
    model,
    tokenizer,
    gamma: float,
    device: str = "cuda",
    dataset_name: str = "lambada",
    dataset_config: str = "test",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
):
    """
    Compute the average entropy of the next‑token distribution over the specified dataset.
    Entropy is defined as -Σ p(x) log p(x) for each step.
    """
    dataset = load_dataset(dataset_name, dataset_config)
    total_entropy = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc=f"Entropy γ={gamma}", leave=False):
        text = example["text"]
        tokenized = tokenizer(text, return_tensors="pt")
        ids = tokenized.input_ids[0]
        if len(ids) < 2:
            continue

        for i in range(1, len(ids)):
            prefix_ids = ids[:i]
            # Conditional logits
            cond_logits = compute_logits(model, tokenizer, prefix_ids, device)
            log_p_cond = torch.log_softmax(cond_logits, dim=-1)

            # Unconditional logits
            uncond_logits = compute_logits(model, tokenizer, prefix_ids, device)
            log_p_uncond = torch.log_softmax(uncond_logits, dim=-1)

            # CFG log‑probability
            new_log_probs = log_p_uncond + gamma * (log_p_cond - log_p_uncond)

            probs = torch.exp(new_log_probs)
            entropy = -torch.sum(probs * new_log_probs).item()
            total_entropy += entropy
            total_tokens += 1

    return total_entropy / total_tokens if total_tokens else 0.0


# --------------------------------------------------------------------------- #
# Chain‑of‑Thought (CoT) evaluation
# --------------------------------------------------------------------------- #
def evaluate_cot(
    model,
    tokenizer,
    gamma: float,
    device: str = "cuda",
    n_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
):
    """
    Evaluate CFG on a small subset of GSM8K using a simple CoT prompt.
    Returns accuracy over the sampled examples.
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    subset = dataset.select(range(min(n_samples, len(dataset))))
    correct = 0
    total = 0

    for example in tqdm(subset, desc=f"CoT γ={gamma}", leave=False):
        question = example["question"]
        answer = example["answer"].strip()
        # Basic CoT prefix
        prompt = f"Question: {question}\nAnswer:"

        generated = generate_cfg(
            model,
            tokenizer,
            prompt,
            gamma=gamma,
            max_new_tokens=120,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        # Check if the expected answer appears as a substring
        if answer in generated:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def evaluate_self_consistency(
    model,
    tokenizer,
    gamma: float,
    device: str = "cuda",
    n_samples: int = 5,
    n_examples: int = 20,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
):
    """
    Self‑Consistency evaluation: sample multiple times per example and
    majority vote on the final answer token.
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    subset = dataset.select(range(min(n_examples, len(dataset))))
    correct = 0
    total = 0

    for example in tqdm(subset, desc=f"Self‑Consist γ={gamma}", leave=False):
        question = example["question"]
        answer = example["answer"].strip()
        prompt = f"Question: {question}\nAnswer:"

        votes = []
        for _ in range(n_samples):
            gen = generate_cfg(
                model,
                tokenizer,
                prompt,
                gamma=gamma,
                max_new_tokens=120,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )
            votes.append(gen)

        # Majority vote
        vote_counts = {}
        for v in votes:
            vote_counts[v] = vote_counts.get(v, 0) + 1
        final_answer = max(vote_counts, key=vote_counts.get)

        if answer in final_answer:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="CFG demo for causal LMs.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-medium",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to store generated outputs",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=60,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top‑k filtering (0 = no filtering)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus (top‑p) filtering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--evaluate_lambada",
        action="store_true",
        help="Run a small LAMBADA evaluation (next‑token prediction)",
    )
    parser.add_argument(
        "--evaluate_perplexity",
        action="store_true",
        help="Run perplexity evaluation on Wikitext‑2",
    )
    parser.add_argument(
        "--evaluate_entropy",
        action="store_true",
        help="Compute average entropy of the logits",
    )
    parser.add_argument(
        "--evaluate_cot",
        action="store_true",
        help="Run Chain‑of‑Thought evaluation on GSM8K",
    )
    parser.add_argument(
        "--evaluate_self_consistency",
        action="store_true",
        help="Run Self‑Consistency evaluation on GSM8K",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Optional negative prompt for CFG (Equation 5)",
    )
    args = parser.parse_args()

    set_random_seed(args.seed)

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # Set pad token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Sample prompts (plain text)
    prompts = [
        "What is the capital of France?",
        "Write a short poem about the sea.",
        "Explain the principle of relativity in simple terms.",
        "Suggest a recipe for a vegetarian lasagna.",
    ]

    gamma_values = [1.0, 1.5, 2.0]

    for prompt in prompts:
        prompt_clean = textwrap.dedent(prompt).strip()
        for gamma in gamma_values:
            output = generate_cfg(
                model,
                tokenizer,
                prompt_clean,
                gamma=gamma,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
                negative_prompt=args.negative_prompt,
            )
            # Save to file
            safe_prompt = "_".join(prompt_clean.split()[:3]).lower()
            filename = out_dir / f"{safe_prompt}_gamma{gamma:.1f}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt_clean}\n")
                f.write(f"γ = {gamma}\n\n")
                f.write(output)
            print(f"Saved: {filename}")

    # ----- Benchmark evaluations -------------------------------------------------
    if args.evaluate_lambada:
        print("\nRunning LAMBADA evaluation...")
        for gamma in gamma_values:
            acc = evaluate_lambada(
                model,
                tokenizer,
                gamma=gamma,
                device=device,
                max_new_tokens=1,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            eval_file = out_dir / f"eval_lambada_gamma{gamma:.1f}.txt"
            with open(eval_file, "w", encoding="utf-8") as f:
                f.write(f"LAMBADA accuracy for γ={gamma:.1f}: {acc*100:.2f}%\n")
            print(f"Saved evaluation to {eval_file}")

    if args.evaluate_perplexity:
        print("\nRunning Wikitext‑2 perplexity evaluation...")
        for gamma in gamma_values:
            ppl = evaluate_perplexity(
                model,
                tokenizer,
                gamma=gamma,
                device=device,
                dataset_name="wikitext",
                dataset_config="wikitext-2",
                max_seq_len=64,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            eval_file = out_dir / f"eval_wikitext_perplexity_gamma{gamma:.1f}.txt"
            with open(eval_file, "w", encoding="utf-8") as f:
                f.write(f"Wikitext‑2 perplexity for γ={gamma:.1f}: {ppl:.2f}\n")
            print(f"Saved evaluation to {eval_file}")

    if args.evaluate_entropy:
        print("\nComputing entropy analysis...")
        for gamma in gamma_values:
            ent = evaluate_entropy(
                model,
                tokenizer,
                gamma=gamma,
                device=device,
                dataset_name="lambada",
                dataset_config="test",
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            eval_file = out_dir / f"entropy_gamma{gamma:.1f}.txt"
            with open(eval_file, "w", encoding="utf-8") as f:
                f.write(f"Average entropy for γ={gamma:.1f}: {ent:.4f}\n")
            print(f"Saved entropy to {eval_file}")

    if args.evaluate_cot:
        print("\nRunning CoT evaluation on GSM8K...")
        for gamma in gamma_values:
            acc = evaluate_cot(
                model,
                tokenizer,
                gamma=gamma,
                device=device,
                n_samples=20,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            eval_file = out_dir / f"cot_accuracy_gamma{gamma:.1f}.txt"
            with open(eval_file, "w", encoding="utf-8") as f:
                f.write(f"CoT accuracy for γ={gamma:.1f}: {acc*100:.2f}%\n")
            print(f"Saved CoT evaluation to {eval_file}")

    if args.evaluate_self_consistency:
        print("\nRunning Self‑Consistency evaluation on GSM8K...")
        for gamma in gamma_values:
            acc = evaluate_self_consistency(
                model,
                tokenizer,
                gamma=gamma,
                device=device,
                n_samples=5,
                n_examples=20,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            eval_file = out_dir / f"selfconsistency_accuracy_gamma{gamma:.1f}.txt"
            with open(eval_file, "w", encoding="utf-8") as f:
                f.write(f"Self‑Consistency accuracy for γ={gamma:.1f}: {acc*100:.2f}%\n")
            print(f"Saved Self‑Consistency evaluation to {eval_file}")

if __name__ == "__main__":
    main()