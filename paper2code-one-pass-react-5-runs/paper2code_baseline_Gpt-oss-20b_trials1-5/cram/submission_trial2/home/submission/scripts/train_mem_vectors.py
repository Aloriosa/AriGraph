import json
import os
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from .utils import load_model_and_tokenizer


def train_memory_for_sample(
    model,
    tokenizer,
    tokens,
    mem_token_id,
    hidden_size,
    device,
    max_steps=500,
    lr=1e-3,
    weight_decay=0.1,
):
    """
    Optimize a single memory vector to encode the given token sequence.
    Returns the trained memory vector and the loss curve.
    """
    # Prepare input and target ids
    input_ids = torch.tensor([mem_token_id] + tokens[:-1], dtype=torch.long).to(device)
    target_ids = torch.tensor(tokens[1:], dtype=torch.long).to(device)

    # Memory vector: one token of shape (1, hidden_size)
    memory_vec = nn.Parameter(torch.randn(1, hidden_size, device=device))
    optimizer = AdamW([memory_vec], lr=lr, weight_decay=weight_decay)
    loss_values = []

    for step in range(max_steps):
        # Get embeddings for the whole input (including the memory placeholder)
        embeddings = model.get_input_embeddings()(input_ids)
        embeddings[0] = memory_vec  # override the placeholder with the trained vector

        outputs = model(inputs_embeds=embeddings, labels=target_ids)
        loss = outputs.loss
        loss_values.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Early stopping if loss is very low (perfect reconstruction)
        if loss.item() < 1e-4:
            break

    return memory_vec.detach(), loss_values


def greedy_decode(
    model,
    tokenizer,
    mem_token_id,
    memory_vec,
    max_length,
    device,
    temperature=1.0,
):
    """
    Greedy decode the sequence that the memory vector encodes.
    """
    generated_ids = []
    mem_emb = memory_vec.to(device)  # (1, hidden_size)

    for _ in range(max_length):
        # Build embeddings for current prefix
        if generated_ids:
            prefix_ids = torch.tensor([mem_token_id] + generated_ids, dtype=torch.long).to(device)
        else:
            prefix_ids = torch.tensor([mem_token_id], dtype=torch.long).to(device)

        embeddings = model.get_input_embeddings()(prefix_ids)
        embeddings[0] = mem_emb  # override memory token embedding

        outputs = model(inputs_embeds=embeddings)
        logits = outputs.logits[:, -1, :] / temperature
        next_id = torch.argmax(logits, dim=-1).item()

        if next_id == tokenizer.eos_token_id:
            break
        generated_ids.append(next_id)

    return generated_ids


def compute_metrics(
    original_tokens, generated_tokens, model, tokenizer, mem_token_id, memory_vec, device
):
    """
    Compute token-level accuracy and cross‑entropy reduction.
    """
    # Token accuracy
    correct = sum(o == g for o, g in zip(original_tokens, generated_tokens))
    accuracy = correct / len(original_tokens)

    # Cross‑entropy of original text (no memory)
    with torch.no_grad():
        input_ids = torch.tensor([original_tokens[:-1]], dtype=torch.long).to(device)
        target_ids = torch.tensor([original_tokens[1:]], dtype=torch.long).to(device)
        outputs = model(input_ids=input_ids, labels=target_ids)
        ce_no_mem = outputs.loss.item()

    # Cross‑entropy with memory
    with torch.no_grad():
        input_ids = torch.tensor([mem_token_id] + original_tokens[:-1], dtype=torch.long).to(device)
        target_ids = torch.tensor([original_tokens[1:]], dtype=torch.long).to(device)
        embeddings = model.get_input_embeddings()(input_ids)
        embeddings[0] = memory_vec.to(device)
        outputs = model(inputs_embeds=embeddings, labels=target_ids)
        ce_with_mem = outputs.loss.item()

    info_gain = ce_no_mem - ce_with_mem
    return accuracy, ce_no_mem, ce_with_mem, info_gain


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model_and_tokenizer(args.model_name, device)
    mem_token_id = tokenizer.convert_tokens_to_ids("[MEM]")
    hidden_size = model.config.hidden_size

    # Load texts
    texts = Path(args.text_file).read_text().strip().splitlines()
    results = []

    for idx, text in enumerate(tqdm(texts, desc="Samples")):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 2:
            continue  # skip too short samples

        # Train memory
        memory_vec, loss_curve = train_memory_for_sample(
            model,
            tokenizer,
            tokens,
            mem_token_id,
            hidden_size,
            device,
            max_steps=args.max_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # Decode
        generated_ids = greedy_decode(
            model,
            tokenizer,
            mem_token_id,
            memory_vec,
            max_length=len(tokens) - 1,
            device=device,
        )

        accuracy, ce_no_mem, ce_with_mem, info_gain = compute_metrics(
            tokens,
            generated_ids,
            model,
            tokenizer,
            mem_token_id,
            memory_vec,
            device,
        )

        results.append(
            {
                "sample_id": idx,
                "text": text,
                "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                "accuracy": accuracy,
                "cross_entropy_no_mem": ce_no_mem,
                "cross_entropy_with_mem": ce_with_mem,
                "info_gain": info_gain,
                "loss_curve": loss_curve,
            }
        )

    # Summary statistics
    avg_acc = sum(r["accuracy"] for r in results) / len(results)
    avg_ce_no = sum(r["cross_entropy_no_mem"] for r in results) / len(results)
    avg_ce_with = sum(r["cross_entropy_with_mem"] for r in results) / len(results)
    avg_info = sum(r["info_gain"] for r in results) / len(results)

    summary = {
        "average_accuracy": avg_acc,
        "average_cross_entropy_no_mem": avg_ce_no,
        "average_cross_entropy_with_mem": avg_ce_with,
        "average_information_gain": avg_info,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as fp:
        json.dump({"samples": results, "summary": summary}, fp, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train memory vectors for each sample.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model name (default: gpt2)",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default="data/sample_texts.txt",
        help="Path to the input text file (one sample per line)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store the results",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum optimisation steps per sample",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for memory optimisation",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay for AdamW",
    )
    args = parser.parse_args()
    main(args)