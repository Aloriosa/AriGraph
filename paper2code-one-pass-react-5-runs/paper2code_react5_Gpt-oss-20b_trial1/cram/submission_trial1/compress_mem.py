#!/usr/bin/env python3
"""
Minimal implementation of the paper's core idea:
train a set of learnable [mem] tokens while freezing the rest of the LLM,
and evaluate the reconstruction quality.
"""

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)


def add_mem_tokens(
    tokenizer: AutoTokenizer, num_mem: int, model: AutoModelForCausalLM
) -> List[int]:
    """Add `[mem]` tokens to tokenizer and resize model embeddings."""
    mem_tokens = [f"[mem{i}]" for i in range(num_mem)]
    tokenizer.add_tokens(mem_tokens)
    model.resize_token_embeddings(len(tokenizer))
    # Return new token IDs
    start_id = tokenizer.vocab_size - num_mem
    return list(range(start_id, tokenizer.vocab_size))


def freeze_model_except_embeddings(model: AutoModelForCausalLM):
    """Freeze all model parameters except the embedding matrix."""
    for p in model.parameters():
        p.requires_grad = False
    model.get_input_embeddings().weight.requires_grad = True


def train_mem_vectors(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    mem_ids: List[int],
    text: str,
    learning_rate: float,
    num_steps: int,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Train the [mem] embeddings and evaluate reconstruction."""
    # Tokenize text (include special tokens)
    text_ids = tokenizer.encode(text, add_special_tokens=True)
    text_ids_tensor = torch.tensor(text_ids, dtype=torch.long, device=device)

    # Prepare training data
    mem_ids_tensor = torch.tensor(mem_ids, dtype=torch.long, device=device)
    input_ids = torch.cat((mem_ids_tensor, text_ids_tensor)).unsqueeze(0)  # batch 1
    labels = input_ids.clone()

    # Optimizer and scheduler
    optimizer = AdamW([model.get_input_embeddings().weight], lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_steps), num_training_steps=num_steps
    )

    # Mask to zero gradients for non‑mem tokens
    vocab_size = tokenizer.vocab_size
    mem_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    mem_mask[mem_ids] = True

    best_loss = float("inf")
    best_step = 0
    start_time = time.time()

    for step in range(1, num_steps + 1):
        model.train()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Zero gradients for non‑mem embeddings
        if model.get_input_embeddings().weight.grad is not None:
            non_mem_grad_mask = ~mem_mask
            model.get_input_embeddings().weight.grad[non_mem_grad_mask] = 0

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_step = step

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            print(
                f"Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | "
                f"Best: {best_loss:.4f} (step {best_step}) | ETA: {elapsed:.1f}s"
            )

        # Early stopping if loss is very small
        if loss.item() < 1e-4:
            print(f"Early stopping at step {step} (loss {loss.item():.4f})")
            break

    # Save embeddings
    emb_path = output_dir / "mem_embeddings.pt"
    torch.save(
        model.get_input_embeddings().weight[mem_ids, :].clone().cpu(),
        str(emb_path),
    )
    print(f"Saved [mem] embeddings to {emb_path}")

    # Evaluation
    metrics = evaluate(
        tokenizer, model, mem_ids, text_ids, device, output_dir
    )
    metrics["loss"] = best_loss
    metrics["best_step"] = best_step
    return metrics


def evaluate(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    mem_ids: List[int],
    text_ids: List[int],
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Generate text from the learned [mem] tokens and compute accuracy."""
    model.eval()
    # Greedy generation
    generated_ids = mem_ids.copy()
    input_ids = torch.tensor([generated_ids], device=device)

    max_gen_len = len(text_ids) + 20  # a bit more than original
    with torch.no_grad():
        for _ in range(max_gen_len):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]  # last token
            next_token = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=device)], dim=1
            )
            if next_token == tokenizer.eos_token_id:
                break

    # Strip mem tokens
    gen_text_ids = generated_ids[len(mem_ids) :]
    # Compute token accuracy
    correct = sum(
        1 for a, b in zip(gen_text_ids, text_ids) if a == b
    )
    accuracy = correct / len(text_ids)

    # Compute cross‑entropy loss on ground truth
    mem_ids_tensor = torch.tensor(mem_ids, dtype=torch.long, device=device)
    text_ids_tensor = torch.tensor(text_ids, dtype=torch.long, device=device)
    input_ids = torch.cat((mem_ids_tensor, text_ids_tensor)).unsqueeze(0)
    labels = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        cross_entropy = outputs.loss.item()

    # Decode generated text
    generated_text = tokenizer.decode(
        gen_text_ids, skip_special_tokens=True
    )

    # Save generated text
    gen_path = output_dir / "generated_text.txt"
    with open(gen_path, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Generated text saved to {gen_path}")

    return {
        "accuracy": accuracy,
        "cross_entropy": cross_entropy,
        "generated_text": generated_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Train [mem] vectors to compress a text.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/gpt-neo-125M",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        required=True,
        help="Path to a text file to compress",
    )
    parser.add_argument(
        "--mem_tokens",
        type=int,
        default=1,
        help="Number of learnable [mem] tokens",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for embedding optimization",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5000,
        help="Maximum number of optimization steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save results",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    # Load text
    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, device_map="auto"
    )
    model.to(device)

    # Add [mem] tokens
    mem_ids = add_mem_tokens(tokenizer, args.mem_tokens, model)
    print(f"Added {args.mem_tokens} [mem] tokens: {mem_ids}")

    # Freeze everything except embeddings
    freeze_model_except_embeddings(model)

    # Train and evaluate
    metrics = train_mem_vectors(
        tokenizer,
        model,
        mem_ids,
        text,
        args.learning_rate,
        args.num_steps,
        device,
        output_dir,
    )

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\nSummary:")
    print(f"  Loss: {metrics['loss']:.4f} (best at step {metrics['best_step']})")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Cross‑Entropy: {metrics['cross_entropy']:.4f}")


if __name__ == "__main__":
    main()