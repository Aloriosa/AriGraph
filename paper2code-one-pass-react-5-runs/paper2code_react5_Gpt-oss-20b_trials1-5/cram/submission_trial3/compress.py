#!/usr/bin/env python3
"""
compress.py – Per‑sample optimisation of trainable memory vectors for a frozen LLM.

Implements the metrics and experiments described in the paper:
  * Decoding capacity (max token length with accuracy ≥ threshold)
  * Token gain
  * Information gain (cross‑entropy reduction)
  * Theoretical capacity bound from Eq. (1)

The script reads a JSON configuration (default `config.json`) that specifies
multiple experiments, each containing a model name, a text file or a flag to
generate random tokens, a set of desired token lengths, number of `[mem]`
vectors, learning rate, etc.
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    """Set deterministic seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_text(path: str, tokenizer: AutoTokenizer, max_tokens: int | None = None
              ) -> str:
    """Read a text file and optionally truncate to a specific number of tokens."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if max_tokens is not None:
        tokens = tokenizer.tokenize(text)
        text = tokenizer.convert_tokens_to_string(tokens[:max_tokens])
    return text


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def theoretical_capacity(model: AutoModelForCausalLM,
                         bits_per_value: int = 16) -> float:
    """Compute upper bound from Eq. (1): d_model * b / log2(|V|)."""
    d_model = model.config.hidden_size
    vocab_size = model.config.vocab_size
    return (d_model * bits_per_value) / np.log2(vocab_size)


def generate_random_token_ids(tokenizer: AutoTokenizer, length: int
                              ) -> torch.Tensor:
    """Generate a random token id sequence (excluding special tokens)."""
    allowed_ids = list(set(range(tokenizer.vocab_size)) - set(tokenizer.all_special_ids))
    # Sample with replacement
    ids = np.random.choice(allowed_ids, size=length, replace=True)
    return torch.tensor(ids, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Core training and evaluation
# --------------------------------------------------------------------------- #
def train_mem_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str | None,
    token_ids: torch.Tensor | None,
    mem_vectors: int,
    lr: float,
    max_steps: int,
    threshold: float,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, float]:
    """
    Optimize a set of trainable `[mem]` embeddings to reconstruct ``text``.
    If ``token_ids`` is provided, ``text`` is ignored.
    Returns the best (detached) embeddings on CPU and the best accuracy.
    """
    model.to(device)
    model.eval()  # freeze LM

    if token_ids is None:
        # Tokenise target text
        token_ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    seq_len = token_ids.shape[0]

    # Initialise trainable embeddings
    d_model = model.config.hidden_size
    mem_embeds = torch.nn.Parameter(
        torch.randn(mem_vectors, d_model, device=device, dtype=dtype)
    )

    optimizer = torch.optim.AdamW([mem_embeds], lr=lr)
    best_acc = 0.0
    best_embeds = None

    for step in tqdm(range(max_steps), desc="Training"):
        optimizer.zero_grad()

        # Build input embeddings: [mem] + text embeddings
        text_embeds = model.get_input_embeddings()(token_ids)  # (seq_len, d_model)
        inputs_embeds = torch.cat([mem_embeds, text_embeds], dim=0).unsqueeze(0)  # (1, seq_len+mem, d_model)

        # Labels: -100 for mem positions (ignored), then token ids
        labels = torch.cat(
            [torch.full((mem_vectors,), -100, dtype=torch.long, device=device), token_ids],
            dim=0,
        ).unsqueeze(0)

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(1, seq_len + mem_vectors, device=device, dtype=torch.long),
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Teacher‑forcing accuracy on text tokens only
        logits = outputs.logits  # (1, seq_len+mem, vocab)
        preds = torch.argmax(logits, dim=-1)  # (1, seq_len+mem)
        preds_text = preds[mem_vectors:, :]
        acc = (preds_text == token_ids).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_embeds = mem_embeds.detach().cpu()

        if acc >= threshold:
            break

    # In case no improvement, still return the final embeddings
    if best_embeds is None:
        best_embeds = mem_embeds.detach().cpu()
    return best_embeds, best_acc


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    token_ids: torch.Tensor,
    mem_embeds: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    """
    Evaluate reconstruction quality and compute metrics.
    """
    model.to(device)
    model.eval()

    seq_len = token_ids.shape[0]
    mem_vectors = mem_embeds.shape[0]

    # ---------- Teacher‑forcing evaluation with [mem] ----------
    mem_embeds = mem_embeds.to(device)
    inputs_embeds = torch.cat(
        [mem_embeds.unsqueeze(0), model.get_input_embeddings()(token_ids)],
        dim=1,
    )
    attention_mask = torch.ones(1, mem_vectors + seq_len, device=device, dtype=torch.long)

    with torch.no_grad():
        outputs_mem = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=token_ids.unsqueeze(0),
        )
        ce_mem = outputs_mem.loss.item()
        logits_mem = outputs_mem.logits  # (1, seq_len+mem, vocab)
        preds_mem = torch.argmax(logits_mem, dim=-1)  # (1, seq_len+mem)
        preds_mem_text = preds_mem[:, mem_vectors:]  # (1, seq_len)
        acc_mem = (preds_mem_text == token_ids.unsqueeze(0)).float().mean().item()

    # ---------- Baseline teacher‑forcing evaluation ----------
    with torch.no_grad():
        outputs_base = model(
            input_ids=token_ids.unsqueeze(0),
            labels=token_ids.unsqueeze(0),
        )
        ce_base = outputs_base.loss.item()
        logits_base = outputs_base.logits
        preds_base = torch.argmax(logits_base, dim=-1)
        acc_base = (preds_base == token_ids.unsqueeze(0)).float().mean().item()

    token_gain = (acc_mem - acc_base) * seq_len
    information_gain = ce_base - ce_mem
    decoding_capacity = seq_len if acc_mem >= threshold else 0

    return {
        "accuracy_with_mem": acc_mem,
        "baseline_accuracy": acc_base,
        "cross_entropy_with_mem": ce_mem,
        "baseline_cross_entropy": ce_base,
        "token_gain": token_gain,
        "information_gain": information_gain,
        "decoding_capacity": decoding_capacity,
    }


# --------------------------------------------------------------------------- #
# Experiment orchestration
# --------------------------------------------------------------------------- #
def run_experiment(exp: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
    """
    Run a single experiment for all requested token lengths and mem‑vector counts.
    Returns a list of result dicts (one per length).
    """
    set_global_seed(seed)
    device = get_device()

    model_name = exp["model"]
    print(f"\n=== Experiment: model={model_name} ===")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    except OSError:
        # Fall back to CPU if GPU is too big
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mem_vectors_cfg = exp["mem_vectors"]
    mem_vectors_list = mem_vectors_cfg if isinstance(mem_vectors_cfg, list) else [mem_vectors_cfg]
    lr = exp["lr"]
    max_steps = exp["max_steps"]
    threshold = exp["threshold"]

    # Load full text once (if needed)
    if exp.get("text_file") and exp["text_file"] != "random":
        full_text = load_text(exp["text_file"], tokenizer)
    else:
        full_text = None

    theoretical_tokens = theoretical_capacity(model)

    results = []

    for mem_vectors in mem_vectors_list:
        for length in exp["lengths"]:
            print(f"\nProcessing length {length} tokens with {mem_vectors} mem vector(s)")
            if exp.get("text_file") == "random":
                token_ids = generate_random_token_ids(tokenizer, length).to(device)
                text = None
            else:
                text = load_text(exp["text_file"], tokenizer, max_tokens=length)
                token_ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)

            # Train mem vectors
            mem_embeds, best_acc = train_mem_vectors(
                model,
                tokenizer,
                text,
                token_ids,
                mem_vectors,
                lr,
                max_steps,
                threshold,
                device,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

            # Evaluate
            eval_res = evaluate(
                model,
                tokenizer,
                token_ids,
                mem_embeds,
                device,
                threshold,
            )

            # Save the best mem vector
            mem_path = Path(f"mem_{model_name.replace('/', '-')}_{length}_{mem_vectors}.pt")
            torch.save(mem_embeds, mem_path)

            result = {
                "model": model_name,
                "mem_vectors": mem_vectors,
                "length": length,
                "theoretical_capacity_tokens": int(round(theoretical_tokens)),
                "decoding_capacity_tokens": eval_res["decoding_capacity"],
                "best_accuracy_with_mem": eval_res["accuracy_with_mem"],
                "baseline_accuracy": eval_res["baseline_accuracy"],
                "token_gain": eval_res["token_gain"],
                "information_gain": eval_res["information_gain"],
                "cross_entropy_with_mem": eval_res["cross_entropy_with_mem"],
                "baseline_cross_entropy": eval_res["baseline_cross_entropy"],
                "compression_ratio": length / mem_vectors,
                "actual_vs_theoretical_ratio": length / theoretical_tokens,
                "mem_vector_path": str(mem_path),
            }
            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Per‑sample optimisation of trainable memory vectors"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output file name",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = cfg.get("seed", 42)
    set_global_seed(seed)

    all_results = []
    for exp in cfg["experiments"]:
        exp_res = run_experiment(exp, seed)
        all_results.extend(exp_res)

    out_file = args.output or cfg.get("output_file", "results.json")
    Path(out_file).write_text(json.dumps(all_results, indent=2))
    print(f"\nAll results written to {out_file}")


if __name__ == "__main__":
    main()