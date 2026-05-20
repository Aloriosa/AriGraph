#!/usr/bin/env python3
"""
compress.py

Implementation of the per‑sample optimisation procedure described in
"Cramming 1568 Tokens into a Single Vector and Back Again".

The script loads a HuggingFace transformer model, adds a set of trainable memory
tokens `[mem]`, optimises only those vectors while keeping the entire model
frozen, and evaluates the compression capacity.

Author: ChatGPT – 2026
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    set_seed,
    logging,
)

# Suppress non‑essential warnings
logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per‑sample compression demo with trainable memory vectors."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model name (default: gpt2)",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        required=True,
        help="Path to the text file to compress",
    )
    parser.add_argument(
        "--num_mem",
        type=int,
        default=1,
        help="Number of trainable memory tokens to prepend",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Maximum number of optimisation steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to store results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Accuracy threshold for decoding capacity metric",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def freeze_all_except_mem(model, mem_ids):
    """Freeze all model parameters except the embedding rows corresponding to mem_ids."""
    for p in model.parameters():
        p.requires_grad = False
    # GPT‑2 uses `model.transformer.wte` for token embeddings
    mem_weight = model.transformer.wte.weight  # shape: (vocab, d_model)
    mem_weight[mem_ids].requires_grad = True


def compute_metrics(model, input_ids, target_ids, device):
    """
    Compute cross‑entropy loss, token accuracy, number of correct tokens,
    and the correctness mask for the given teacher‑forced input/target pair.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]          # shape: [1, seq_len-1, vocab]
        labels = target_ids[:, 1:]                  # shape: [1, seq_len-1]
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).item()

        preds = torch.argmax(logits, dim=-1)          # [1, seq_len-1]
        mask = labels != -100
        correct = (preds == labels) & mask
        correct_count = correct.sum().item()
        total = mask.sum().item()
        acc = correct_count / total if total > 0 else 0.0

    return loss, acc, correct_count, correct


def train_mem_vectors(
    model, input_ids, target_ids, max_steps, lr, device
):
    """Train only the memory embeddings to predict the target sequence."""
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    best_loss = float("inf")
    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        loss, _, _, _ = compute_metrics(model, input_ids, target_ids, device)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1 or step == max_steps:
            print(f"Step {step:4d} | Loss: {loss:.6f}")

        if loss < best_loss:
            best_loss = loss

        if loss < 1e-6:
            print(f"Converged at step {step}")
            break

    return best_loss


def compute_theoretical_bound(d_model, vocab_size, bit_precision=16):
    """
    Compute the theoretical capacity bound L ≤ d_model × b / log₂|V|.
    """
    return int(d_model * bit_precision / math.log2(vocab_size))


def compute_decoding_capacity(correct_mask, threshold):
    """
    Determine the longest prefix length L for which the accuracy
    (correct tokens / prefix length) is above the given threshold.
    `correct_mask` is a 1‑D tensor of booleans for each original token.
    """
    if len(correct_mask) == 0:
        return 0
    cumulative_correct = torch.cumsum(correct_mask, dim=0)
    positions = torch.arange(1, len(correct_mask) + 1, device=correct_mask.device)
    acc_prefix = cumulative_correct.float() / positions
    # Find all positions where accuracy >= threshold
    valid = (acc_prefix >= threshold)
    if not valid.any():
        return 0
    Lmax = int(valid.nonzero(as_tuple=False)[-1].item() + 1)
    return Lmax


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer & model
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)

    # Add memory tokens
    mem_tokens = [f"[MEM_{i}]" for i in range(args.num_mem)]
    new_tokens = [t for t in mem_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {len(new_tokens)} new memory tokens.")
    mem_ids = tokenizer.convert_tokens_to_ids(mem_tokens)

    # Freeze all weights except the memory embeddings
    freeze_all_except_mem(model, mem_ids)

    # Read the text to compress
    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError("Input text file is empty.")

    # Tokenise the text
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) == 0:
        raise ValueError("Tokenised text is empty.")
    seq_len = len(input_ids)

    # ---------- Baseline metrics (no memory) ----------
    # Input: original tokens
    orig_input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    # Target: shift by 1, ignore first token
    orig_target_ids = torch.full_like(orig_input_ids, -100, device=device)
    orig_target_ids[:, 1:] = orig_input_ids[:, 1:]

    baseline_loss, baseline_acc, baseline_correct, _ = compute_metrics(
        model, orig_input_ids, orig_target_ids, device
    )
    print("\n--- Baseline (no memory) ---")
    print(f"Cross‑entropy loss: {baseline_loss:.6f}")
    print(f"Token accuracy: {baseline_acc * 100:.2f}%")

    # ---------- Prepare data for memory optimisation ----------
    # Input: [mem] + original tokens
    mem_tensor = torch.tensor([mem_ids], dtype=torch.long, device=device)
    input_seq = torch.cat([mem_tensor, orig_input_ids], dim=1)

    # Target: shift by 1, ignore first mem tokens
    target_seq = torch.full_like(input_seq, -100, device=device)
    target_seq[:, len(mem_ids) :] = orig_input_ids

    # ---------- Train memory vectors ----------
    print("\n--- Training memory vectors ---")
    train_loss = train_mem_vectors(
        model, input_seq, target_seq, args.max_steps, args.lr, device
    )

    # ---------- Evaluate memory‑conditioned metrics ----------
    mem_loss, mem_acc, mem_correct, correct_mask = compute_metrics(
        model, input_seq, target_seq, device
    )
    print("\n--- With memory ---")
    print(f"Cross‑entropy loss: {mem_loss:.6f}")
    print(f"Token accuracy: {mem_acc * 100:.2f}%")

    # Token gain
    token_gain = mem_correct - baseline_correct

    # Information gain (cross‑entropy reduction)
    info_gain = baseline_loss - mem_loss

    # Theoretical capacity bound
    d_model = model.config.n_embd
    vocab_size = tokenizer.vocab_size
    theoretical_bound = compute_theoretical_bound(d_model, vocab_size)

    # Decoding capacity: longest prefix length with accuracy >= threshold
    # The correct_mask tensor includes predictions for all tokens after the first.
    # Slice out only the part corresponding to original tokens.
    correct_mask_orig = correct_mask[:seq_len]
    decoding_capacity = compute_decoding_capacity(correct_mask_orig, args.threshold)

    # ---------- Generate reconstructed text ----------
    generated_ids = []
    cur_input = mem_tensor.clone()
    model.eval()
    with torch.no_grad():
        for _ in range(seq_len):
            outputs = model(cur_input)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_token)
            cur_input = torch.tensor([[next_token]], dtype=torch.long, device=device)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ---------- Save results ----------
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "original.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.output_dir, "generated.txt"), "w", encoding="utf-8") as f:
        f.write(generated_text)

    report = {
        "model_name": args.model_name,
        "num_mem": args.num_mem,
        "seq_len": seq_len,
        "baseline_loss": baseline_loss,
        "baseline_acc": baseline_acc,
        "train_loss": train_loss,
        "mem_loss": mem_loss,
        "mem_acc": mem_acc,
        "token_gain": token_gain,
        "info_gain": info_gain,
        "theoretical_bound": theoretical_bound,
        "decoding_capacity": decoding_capacity,
    }
    with open(os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n=== Summary ===")
    print(f"Original text:\n{text}\n")
    print(f"Reconstructed text:\n{generated_text}\n")
    print(f"Token accuracy: {mem_acc * 100:.2f}%")
    print(f"Token gain: {token_gain}")
    print(f"Information gain (loss reduction): {info_gain:.6f}")
    print(f"Theoretical capacity bound: {theoretical_bound} tokens")
    print(f"Decoding capacity (longest prefix ≥ {args.threshold}): {decoding_capacity}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()