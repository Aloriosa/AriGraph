#!/usr/bin/env python3
"""
APT: Adaptive Pruning & Tuning – Minimal reproducible experiment
Fine‑tunes a DistilBERT base model on SST‑2 with dynamic LoRA adapters,
structured output‑dimension pruning, adaptive rank growth, and self‑distillation.
"""

import argparse
import os
import time
import random
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)

from apt_modules import LoRAMaskedLinear

# --------------------------- Configuration ---------------------------

parser = argparse.ArgumentParser(description="APT Reproduction")
parser.add_argument("--mode", choices=["apt", "lora", "prune"], default="apt",
                    help="Training mode: apt (full APT), lora (LoRA only), prune (pruning only)")
parser.add_argument("--model_name", default="distilbert-base-uncased",
                    help="Pretrained model name")
parser.add_argument("--task", default="sst2", help="GLUE task")
parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--target_sparsity", type=float, default=0.6,
                    help="Final target sparsity for pruning (0-1)")
parser.add_argument("--initial_rank", type=int, default=4, help="Initial LoRA rank")
parser.add_argument("--max_rank", type=int, default=16, help="Maximum LoRA rank")
parser.add_argument("--distill_weight", type=float, default=0.5,
                    help="Weight of self‑distillation loss (will anneal to 1)")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------------- Helper functions ---------------------------

def replace_linear_with_lora_masked(model: nn.Module, rank: int):
    """
    Recursively replace every nn.Linear in the model with LoRAMaskedLinear.
    Keeps a list of all replaced layers for later access.
    """
    lora_layers = []

    def _replace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                lora_linear = LoRAMaskedLinear(child, rank=rank)
                setattr(module, name, lora_linear)
                lora_layers.append(lora_linear)
            else:
                _replace(child)

    _replace(model)
    return lora_layers

def compute_layer_saliences(lora_layers):
    """
    After a backward pass, compute salience (sum abs grads + sqrt kurtosis)
    for each layer's output dimensions.
    Returns a list of tensors of shape (out_features,).
    """
    saliences = []
    for layer in lora_layers:
        sal = layer.salience()  # shape (out_features,)
        saliences.append(sal)
    return saliences

def prune_layers(lora_layers, target_sparsity):
    """
    For each layer, prune output dimensions with lowest salience density
    until the overall target sparsity is reached.
    """
    # Total number of output dims across all layers
    total_dims = sum(layer.W_orig.shape[0] for layer in lora_layers)
    keep_dims = int(total_dims * (1 - target_sparsity))

    # Compute salience density for each dimension
    all_saliences = torch.cat([sal for sal in compute_layer_saliences(lora_layers)])
    # Density = salience / (total_params of that dim)
    # For simplicity, use equal weight per dim (since we only prune outputs)
    density = all_saliences  # higher is more important

    # Sort dimensions by density descending
    sorted_idx = torch.argsort(density, descending=True)

    # Build mask for each layer
    masks = []
    idx = 0
    for layer in lora_layers:
        out_dim = layer.W_orig.shape[0]
        layer_mask = torch.ones(out_dim, dtype=torch.bool, device=layer.W_orig.device)
        masks.append(layer_mask)
    remaining = keep_dims

    # Greedy selection of dimensions to keep
    for dim_idx in sorted_idx:
        if remaining <= 0:
            break
        # Find which layer this dimension belongs to
        cum = 0
        for layer_idx, layer in enumerate(lora_layers):
            dim_start = cum
            dim_end = cum + layer.W_orig.shape[0]
            if dim_start <= dim_idx < dim_end:
                local_idx = dim_idx - dim_start
                masks[layer_idx][local_idx] = True
                remaining -= 1
                break
            cum = dim_end

    # Apply masks
    for layer, mask in zip(lora_layers, masks):
        layer.mask_prune(mask)

def increase_rank_for_salient_layers(lora_layers, top_frac=0.5, increment=4):
    """
    Increase LoRA rank by `increment` for the top fraction of layers
    ranked by their total salience. Layers already at MAX_RANK are skipped.
    """
    # Compute total salience per layer
    layer_salience = torch.tensor([sal.sum() for sal in compute_layer_saliences(lora_layers)],
                                  device=lora_layers[0].device)
    # Rank layers
    sorted_idx = torch.argsort(layer_salience, descending=True)
    top_n = max(1, int(len(lora_layers) * top_frac))
    for idx in sorted_idx[:top_n]:
        layer = lora_layers[idx]
        if layer.rank < args.max_rank:
            new_rank = min(args.max_rank, layer.rank + increment)
            layer.increase_rank(new_rank)

def evaluate(model, tokenizer, dataset, batch_size=32):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs = tokenizer(batch["sentence"], truncation=True, padding=True,
                               return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch["label"]).sum().item()
            total += len(batch["label"])
    return correct / total

# --------------------------- Main script ---------------------------

def main():
    # Load dataset
    dataset = load_dataset("glue", args.task)
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    # Build model
    base_model = DistilBertModel.from_pretrained(args.model_name)
    lora_layers = replace_linear_with_lora_masked(base_model, rank=args.initial_rank)
    model = base_model.to(DEVICE)

    # Classification head
    classifier = nn.Linear(model.config.dim, 2).to(DEVICE)  # SST-2 binary

    # Optimizer
    params = lora_layers + [classifier]
    if args.mode != "prune":  # prune-only mode does not update LoRA params
        optimizer = AdamW(params, lr=args.lr)
    else:
        optimizer = AdamW([classifier], lr=args.lr)  # only classifier is trained
    total_steps = args.epochs * len(dataset["train"]) // args.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # Logging structures
    log_dict = {
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "target_sparsity": args.target_sparsity,
        "initial_rank": args.initial_rank,
        "max_rank": args.max_rank,
        "distill_weight": args.distill_weight,
        "time_min": 0.0,
        "peak_mem": 0.0,
        "val_acc": 0.0,
        "inference_time": 0.0,
    }

    # Self‑distillation teacher (only for APT)
    if args.mode == "apt":
        teacher = model  # shared parameters
        teacher.eval()
    else:
        teacher = None

    # Training loop
    start_time = time.time()
    max_mem = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)

        epoch_loss = 0.0
        for batch in loader:
            inputs = tokenizer(batch["sentence"], truncation=True, padding=True,
                               return_tensors="pt").to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = classifier(outputs.last_hidden_state[:, -1, :])  # CLS token
            loss = F.cross_entropy(logits, labels)

            # Self‑distillation loss for APT
            if args.mode == "apt" and teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher(**inputs).last_hidden_state[:, -1, :]
                distill_loss = F.mse_loss(outputs.last_hidden_state[:, -1, :], teacher_out)
                mu = epoch / args.epochs  # anneal to 1 over epochs
                loss = (1 - mu) * loss + mu * args.distill_weight * distill_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            # Track peak memory
            cur_mem = torch.cuda.max_memory_allocated(DEVICE)
            if cur_mem > max_mem:
                max_mem = cur_mem

        # After backward, update pruning and rank
        target_sparsity = args.target_sparsity * (epoch / args.epochs)
        prune_layers(lora_layers, target_sparsity)
        if args.mode == "apt":
            increase_rank_for_salient_layers(lora_layers)

        # Evaluate
        acc = evaluate(model, tokenizer, dataset["validation"])
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch} | Loss: {epoch_loss/len(loader):.4f} | "
              f"Val Acc: {acc*100:.2f}% | Time: {epoch_time:.2f}s | "
              f"Peak Mem: {max_mem/1024**2:.2f} MB")

    total_time = time.time() - start_time
    log_dict["time_min"] = total_time / 60
    log_dict["peak_mem"] = max_mem / 1024**2
    log_dict["val_acc"] = acc

    # Final evaluation
    final_acc = evaluate(model, tokenizer, dataset["validation"])
    log_dict["val_acc"] = final_acc

    # Inference speed test
    loader = DataLoader(dataset["validation"], batch_size=args.batch_size)
    start = time.time()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = tokenizer(batch["sentence"], truncation=True, padding=True,
                               return_tensors="pt").to(DEVICE)
            _ = model(**inputs)
    inference_time = time.time() - start
    log_dict["inference_time"] = inference_time

    # Save checkpoint
    output_dir = Path("apt_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "classifier_state": classifier.state_dict(),
        "tokenizer": tokenizer,
        "lora_layers": [layer.state_dict() for layer in lora_layers],
    }, str(output_dir))

    # Log to JSON for summary script
    log_file = Path(f"{args.mode}_log.json")
    log_file.write_text(json.dumps(log_dict, indent=2))

    print("\nTraining completed.")
    print(f"Total time: {total_time/60:.2f} minutes.")
    print(f"Peak GPU memory used: {max_mem/1024**2:.2f} MB")
    print(f"Final validation accuracy: {final_acc*100:.2f}%")
    print(f"Inference time on validation set: {inference_time:.2f}s "
          f"({inference_time/len(loader):.3f}s per batch)")

if __name__ == "__main__":
    main()