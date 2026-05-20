#!/usr/bin/env python3
"""
Minimal implementation of per‑sample optimization of one or more [mem] vectors
to reconstruct a short text using a frozen causal language model.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def baseline_metrics(model, tokenizer, text: str, device: torch.device):
    """
    Compute baseline LM metrics (accuracy, cross‑entropy) on the text without any [mem] vectors.
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)  # [1, N]
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.item()  # cross‑entropy loss (per token, averaged)
        logits = outputs.logits[0, :, :]  # [N, vocab]
        preds = logits.argmax(dim=-1)  # [N]
        correct = (preds == labels[0]).sum().item()
        accuracy = correct / labels[0].size(0)

    return {"accuracy": accuracy, "cross_entropy": loss}


def train(
    model,
    tokenizer,
    text: str,
    mem_vectors: int,
    steps: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    threshold: float = 0.99,
):
    """
    Train a set of [mem] vectors to encode the given text.
    Returns a dict with final metrics and the learned vectors.
    """
    # Tokenise the input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)  # shape [1, N]
    labels = input_ids.clone()

    # Create [mem] embeddings: one vector per token (mem_vectors=1 or >1)
    hidden_size = model.config.hidden_size
    mem_embeds = torch.nn.Parameter(
        torch.randn(mem_vectors, hidden_size, device=device, requires_grad=True)
    )

    optimizer = torch.optim.AdamW([mem_embeds], lr=lr)

    # Prepare labels with ignore_index for the [mem] positions
    ignore_idx = -100
    full_labels = torch.full(
        (1, mem_vectors + input_ids.size(1)), ignore_idx, dtype=torch.long, device=device
    )
    full_labels[0, mem_vectors :] = labels[0]

    best_loss = float("inf")
    best_step = 0

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        # Build inputs_embeds: [mem] embeddings + token embeddings
        token_embeds = model.get_input_embeddings()(input_ids)  # [1, N, d]
        inputs_embeds = torch.cat([mem_embeds.unsqueeze(0), token_embeds], dim=1)

        outputs = model(inputs_embeds=inputs_embeds, labels=full_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_step = step

        if step % 200 == 0 or step == 1:
            print(f"Step {step}/{steps} | Loss: {loss.item():.4f}")

    # After training, evaluate token accuracy
    with torch.no_grad():
        predictions = model(inputs_embeds=inputs_embeds).logits  # [1, L, vocab]
        preds = predictions[0, mem_vectors :, :].argmax(dim=-1)  # [N]
        targets = labels[0]  # [N]
        correct = (preds == targets).sum().item()
        accuracy = correct / targets.size(0)

        # Cross‑entropy on the text (after mem)
        log_probs = F.log_softmax(predictions[0, mem_vectors :, :], dim=-1)
        ce = -log_probs[torch.arange(targets.size(0)), targets].mean().item()

    # Compute decoding capacity: true if accuracy >= threshold
    decoding_capacity = input_ids.size(1) if accuracy >= threshold else 0

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "vectors.npy", mem_embeds.detach().cpu().numpy())

    decoded_ids = preds.cpu().numpy()
    decoded_text = tokenizer.decode(decoded_ids, skip_special_tokens=True)
    with open(output_dir / "decoded.txt", "w", encoding="utf-8") as f:
        f.write(decoded_text)

    metrics = {
        "steps": steps,
        "final_loss": loss.item(),
        "best_loss": best_loss,
        "best_step": best_step,
        "token_accuracy": accuracy,
        "cross_entropy": ce,
        "input_length": input_ids.size(1),
        "mem_vectors": mem_vectors,
        "decoding_capacity": decoding_capacity,
        "threshold": threshold,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compress a short text into one or more [mem] vectors."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilgpt2",
        help="HuggingFace model id (must be a causal LM).",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default="",
        help="Path to input text (single file). Use --text_dir for multiple files.",
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        default="",
        help="Directory containing multiple text files (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory.",
    )
    parser.add_argument(
        "--mem_vectors",
        type=int,
        default=1,
        help="Number of [mem] vectors to optimise.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of optimisation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Accuracy threshold for decoding capacity.",
    )
    args = parser.parse_args()

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Prepare list of texts
    texts = []
    if args.text_dir:
        for p in Path(args.text_dir).glob("*.txt"):
            texts.append((p.name, load_text(str(p))))
    elif args.text_file:
        texts.append((Path(args.text_file).name, load_text(args.text_file)))
    else:
        raise ValueError("Either --text_file or --text_dir must be specified")

    overall_metrics = {"texts": {}}

    for fname, txt in texts:
        print(f"\n=== Processing {fname} ({len(txt.split())} tokens) ===")
        metrics = train(
            model,
            tokenizer,
            txt,
            mem_vectors=args.mem_vectors,
            steps=args.steps,
            lr=args.lr,
            device=device,
            output_dir=Path(args.output_dir) / fname.replace(".txt", ""),
            threshold=args.threshold,
        )

        # Baseline LM metrics
        baseline = baseline_metrics(model, tokenizer, txt, device)

        # Token gain & information gain
        token_gain = metrics["token_accuracy"] * metrics["input_length"] - baseline["accuracy"] * metrics["input_length"]
        info_gain = baseline["cross_entropy"] - metrics["cross_entropy"]

        overall_metrics["texts"][fname] = {
            "compressed": metrics,
            "baseline": baseline,
            "token_gain": token_gain,
            "information_gain": info_gain,
        }

    # Aggregate metrics
    overall_metrics["summary"] = {
        "num_texts": len(texts),
        "average_token_accuracy": np.mean(
            [t["compressed"]["token_accuracy"] for t in overall_metrics["texts"].values()]
        ),
        "average_token_gain": np.mean(
            [t["token_gain"] for t in overall_metrics["texts"].values()]
        ),
        "average_information_gain": np.mean(
            [t["information_gain"] for t in overall_metrics["texts"].values()]
        ),
    }

    # Save aggregated metrics
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    print("\n=== All metrics written to", Path(args.output_dir) / "metrics.json", "===")


if __name__ == "__main__":
    main()