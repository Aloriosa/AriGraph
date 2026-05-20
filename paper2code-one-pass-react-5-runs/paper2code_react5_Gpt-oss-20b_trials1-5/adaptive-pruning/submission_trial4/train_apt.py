import argparse
import os
import random
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, AdamW, get_linear_schedule_with_warmup)

from datasets import load_dataset
from accelerate import Accelerator

from models.apt_adapter import APTLinear
import utils

# --------------------------------------------------------------------------- #
#  Hyper‑parameters & configuration
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "model_name": "roberta-base",
    "task": "sst2",
    "epochs": 3,
    "batch_size": 32,
    "lr": 2e-5,
    "seed": 42,
    "prune_rate": 0.60,      # target sparsity 60%
    "initial_rank": 8,
    "max_rank": 32,
    "rank_increase_every": 1,   # epochs
    "save_dir": "outputs",
}

# --------------------------------------------------------------------------- #
#  Utility functions
# --------------------------------------------------------------------------- #
def replace_linear_with_apt(module, device, initial_rank):
    """
    Recursively replace nn.Linear modules with APTLinear.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_module = APTLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                rank=initial_rank,
                device=device
            )
            # Copy weight and bias
            new_module.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new_module.bias.data = child.bias.data.clone()
            setattr(module, name, new_module)
        else:
            replace_linear_with_apt(child, device, initial_rank)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)
            outputs = model(**inputs).logits
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# --------------------------------------------------------------------------- #
#  Main training loop
# --------------------------------------------------------------------------- #
def main(args):
    utils.set_seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device

    # Load dataset
    dataset = load_dataset("glue", args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch['sentence'], truncation=True, padding=False)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    train_loader = DataLoader(
        dataset["train"], shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        batch_size=args.batch_size
    )
    dev_loader = DataLoader(
        dataset["validation"],
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        batch_size=args.batch_size
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        output_hidden_states=False
    ).to(device)

    # Inject APT adapters
    replace_linear_with_apt(model, device, args.initial_rank)

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )

    model, optimizer, train_loader, dev_loader = accelerator.prepare(
        model, optimizer, train_loader, dev_loader
    )

    # Training
    best_acc = 0.0
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} Step {step+1} Loss {loss.item():.4f}")

        # Prune neurons after each epoch
        prune_ratio = (args.prune_rate - 0.0) / args.epochs  # linear schedule
        for name, module in model.named_modules():
            if isinstance(module, APTLinear):
                module.prune_neurons(prune_ratio)

        # Increase LoRA rank periodically
        if (epoch + 1) % args.rank_increase_every == 0:
            for name, module in model.named_modules():
                if isinstance(module, APTLinear):
                    module.increase_rank(min(module.rank + args.initial_rank,
                                            args.max_rank))

        # Evaluation
        acc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1} Dev Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            # Save checkpoint
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            accelerator.save_state(Path(args.save_dir) / "best_checkpoint.pt")

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed/60:.2f} minutes.")
    print(f"Best Dev Accuracy: {best_acc:.4f}")

    # Final evaluation on dev set
    acc = evaluate(model, dev_loader, device)
    print(f"Final Dev Accuracy: {acc:.4f}")

    # Report memory usage
    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak GPU memory (MB): {max_mem:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APT Training Demo")
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    main(args)