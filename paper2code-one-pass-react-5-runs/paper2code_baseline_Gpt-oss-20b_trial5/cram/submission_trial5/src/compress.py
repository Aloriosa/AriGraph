#!/usr/bin/env python3
"""
Simple per‑sample optimisation of a single memory vector.
The script trains a frozen causal language model (default GPT‑2) to
encode a short sentence into a single learnable token `[MEM]`.
After training, the model is used to decode the sentence
losslessly.  Results are written to a JSON file.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def freeze_model_except(model: nn.Module, param_names: list[str]) -> None:
    """Freeze all parameters in the model except those whose names contain
    any of the strings in param_names."""
    for name, param in model.named_parameters():
        if any(pn in name for pn in param_names):
            param.requires_grad = True
        else:
            param.requires_grad = False

# --------------------------------------------------------------------------- #
# Core training / decoding logic
# --------------------------------------------------------------------------- #

def train_memory_vector(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    mem_token: str = "[MEM]",
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    max_steps: int = 200,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train a single `[MEM]` token to encode `text` with a frozen LM.

    Returns a dictionary with training history and decoded text.
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Tokenise the sentence
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"].squeeze(0).to(device)  # [seq_len]
    seq_len = input_ids.shape[0]
    vocab_size = tokenizer.vocab_size

    # Add a new token to the tokenizer and resize model embeddings
    tokenizer.add_tokens([mem_token])
    model.resize_token_embeddings(len(tokenizer))
    mem_id = tokenizer.convert_tokens_to_ids(mem_token)

    # Freeze all model parameters except the new embedding
    freeze_model_except(model, ["embed_tokens.weight"])
    # Ensure only the new embedding is trainable
    for param in model.parameters():
        if param is not None:
            param.requires_grad = False
    # Make the new embedding trainable
    mem_emb = model.get_input_embeddings().weight[mem_id]
    mem_emb.requires_grad = True

    optimizer = AdamW([mem_emb], lr=lr, weight_decay=weight_decay)
    loss_fct = nn.CrossEntropyLoss()

    history = {
        "steps": [],
        "loss": [],
        "accuracy": [],
    }

    best_acc = 0.0
    best_mem = mem_emb.detach().clone()

    # Training loop
    for step in tqdm(range(1, max_steps + 1), disable=not verbose):
        # Build input: prepend mem_id token
        cur_input = torch.cat([torch.tensor([[mem_id]], device=device), input_ids.unsqueeze(0)], dim=1)

        # Labels: original tokens (no mem token)
        labels = input_ids.unsqueeze(0)

        optimizer.zero_grad()
        outputs = model(cur_input, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Record stats
        with torch.no_grad():
            logits = outputs.logits  # [1, seq_len+1, vocab]
            preds = logits.argmax(-1).squeeze(0)[1:]  # drop mem token
            acc = (preds == labels.squeeze(0)).float().mean().item()

        history["steps"].append(step)
        history["loss"].append(loss.item())
        history["accuracy"].append(acc)

        if acc > best_acc:
            best_acc = acc
            best_mem = mem_emb.detach().clone()

        if acc >= 1.0:
            if verbose:
                print(f"Reached perfect accuracy at step {step}.")
            break

    # Load best embedding
    mem_emb.data = best_mem

    # --------------------------------------------------------------------- #
    # Decoding
    # --------------------------------------------------------------------- #
    model.eval()
    generated_ids = [mem_id]
    cur_input = torch.tensor([[mem_id]], device=device)
    with torch.no_grad():
        for _ in range(seq_len):
            logits = model(cur_input).logits
            next_id = logits[0, -1].argmax(-1).item()
            generated_ids.append(next_id)
            cur_input = torch.cat([cur_input, torch.tensor([[next_id]], device=device)], dim=1)

    decoded_text = tokenizer.decode(generated_ids[1:], skip_special_tokens=True)

    return {
        "text": text,
        "generated_text": decoded_text,
        "memory_token_id": mem_id,
        "steps_trained": step,
        "final_accuracy": acc,
        "final_loss": loss.item(),
        "history": history,
    }

# --------------------------------------------------------------------------- #
# Argument parsing & main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single memory vector to encode a sentence.")
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name")
    parser.add_argument("--text", type=str, required=True, help="Text to compress")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum optimisation steps")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")

    args = parser.parse_args()

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)

    result = train_memory_vector(
        model=model,
        tokenizer=tokenizer,
        text=args.text,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Final Result ===")
    print(f"Original text   : {result['text']}")
    print(f"Generated text  : {result['generated_text']}")
    print(f"Accuracy        : {result['final_accuracy']:.4f}")
    print(f"Loss            : {result['final_loss']:.4f}")
    print(f"Steps trained   : {result['steps_trained']}")
    print(f"Results written to {out_path}")

if __name__ == "__main__":
    main()