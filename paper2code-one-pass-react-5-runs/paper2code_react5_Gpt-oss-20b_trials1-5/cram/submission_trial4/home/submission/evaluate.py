#!/usr/bin/env python3
"""
Utility script that loads the trained [mem] vectors and performs a full decoding,
computing token‑level accuracy and cross‑entropy for each text.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate(
    model,
    tokenizer,
    mem_vectors: np.ndarray,
    text: str,
    device: torch.device,
):
    """
    Given a trained mem_vectors array (shape [k, d]) and the original text,
    returns accuracy and cross_entropy of the LM when decoding from mem.
    """
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    input_ids = inputs["input_ids"].to(device)
    labels = input_ids.clone()

    # Build inputs_embeds
    k = mem_vectors.shape[0]
    mem_embeds = torch.tensor(mem_vectors, dtype=torch.float32, device=device)

    token_embeds = model.get_input_embeddings()(input_ids)  # [1, N, d]
    inputs_embeds = torch.cat([mem_embeds.unsqueeze(0), token_embeds], dim=1)

    with torch.no_grad():
        logits = model(inputs_embeds=inputs_embeds).logits  # [1, L, vocab]
        preds = logits[0, k :, :].argmax(dim=-1)  # [N]
        correct = (preds == labels[0]).sum().item()
        accuracy = correct / labels[0].size(0)

        log_probs = F.log_softmax(logits[0, k :, :], dim=-1)
        ce = -log_probs[torch.arange(labels[0].size(0)), labels[0]].mean().item()

    return {"accuracy": accuracy, "cross_entropy": ce}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a saved mem vector set.")
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--vector_file", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    mem_vectors = np.load(args.vector_file)
    text = Path(args.text_file).read_text(encoding="utf-8")

    metrics = evaluate(model, tokenizer, mem_vectors, text, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()