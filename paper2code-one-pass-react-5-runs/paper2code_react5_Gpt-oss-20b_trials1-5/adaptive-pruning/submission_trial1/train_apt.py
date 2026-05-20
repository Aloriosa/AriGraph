"""
Minimal APT implementation on BERT‑base fine‑tuning for SST‑2.
This script demonstrates the core ideas from the APT paper:
  • Outlier‑aware salience scoring for pruning heads
  • Adaptive rank expansion of LoRA‑style adapters
  • Self‑distillation using a moving teacher copy
"""

import copy
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    logging,
    set_seed,
)

logging.set_verbosity_error()
set_seed(42)

# --------------------------------------------------------------------------- #
# 1. Helper classes: APTAdapter (LoRA style) and pruning utilities
# --------------------------------------------------------------------------- #
class APTAdapter(nn.Module):
    """
    LoRA‑style adapter with dynamic rank expansion.
    Applied to a linear layer:  y = W x + (B @ A) x
    """
    def __init__(self, in_features: int, out_features: int, rank: int, scaling: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling

        # Low‑rank matrices
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x):
        # Standard linear projection
        out = F.linear(x, self.weight)  # weight is added later

        # LoRA contribution
        lora = (self.B @ self.A @ x.transpose(-1, -2)).transpose(-1, -2)
        return out + self.scaling * lora

    @property
    def weight(self):
        """
        Exposed weight property so the calling code can replace the original weight.
        """
        return self._weight

    @weight.setter
    def weight(self, w):
        self._weight = w

    def expand_rank(self, new_rank: int):
        """Add new columns to A and rows to B, initialized to zero."""
        assert new_rank >= self.rank
        if new_rank == self.rank:
            return
        # Expand A
        extra_A = torch.zeros(new_rank - self.rank, self.in_features, device=self.A.device)
        self.A = nn.Parameter(torch.cat([self.A, extra_A], dim=0))
        # Expand B
        extra_B = torch.zeros(self.out_features, new_rank - self.rank, device=self.B.device)
        self.B = nn.Parameter(torch.cat([self.B, extra_B], dim=1))
        self.rank = new_rank


def replace_with_apt_adapter(module: nn.Module, rank: int, scaling: float = 1.0) -> None:
    """
    Recursively replace all nn.Linear layers in a module with APTAdapters.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            in_f, out_f = child.in_features, child.out_features
            adapter = APTAdapter(in_f, out_f, rank, scaling)
            adapter.weight = child.weight
            setattr(module, name, adapter)
        else:
            replace_with_apt_adapter(child, rank, scaling)


def get_attention_head_salience(model: nn.Module, head_mask: torch.Tensor, grad_cache: dict) -> torch.Tensor:
    """
    Compute salience for each head as the sum of |grad * weight| for the query weights.
    Only heads that are currently active (mask==1) are considered.
    """
    salience = torch.zeros(head_mask.size(0), device=head_mask.device)
    # The BERT attention layers are in the form `self.attention.self.query` etc.
    # Each query linear layer has shape (hidden_size, hidden_size) and is split into heads.
    # For simplicity, we compute salience over the entire weight matrix and then split.
    for name, module in model.named_modules():
        if "attention.self.query" in name and isinstance(module, APTAdapter):
            # Shape: (hidden_size, hidden_size)
            weight = module.weight  # original weight
            grad = grad_cache.get(name, torch.zeros_like(weight))
            head_dim = weight.size(1) // head_mask.size(0)
            # Flatten weight and grad for salience
            flat_weight = weight.view(head_mask.size(0), head_dim, weight.size(0))
            flat_grad = grad.view(head_mask.size(0), head_dim, grad.size(0))
            # Compute |grad * weight| sum over dims
            head_sal = torch.sum(torch.abs(flat_grad * flat_weight), dim=(1, 2))
            salience += head_sal
    return salience


# --------------------------------------------------------------------------- #
# 2. Dataset and DataCollator
# --------------------------------------------------------------------------- #
@dataclass
class SST2Config:
    dataset_name: str = "glue"
    subset_name: str = "sst2"
    max_length: int = 128
    batch_size: int = 32
    num_epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    prune_interval: int = 1  # prune after every epoch
    prune_fraction: float = 0.15  # fraction of lowest‑salience heads to prune
    rank_init: int = 8
    rank_max: int = 32
    rank_increase_per_epoch: int = 8
    scaling: float = 1.0


def load_sst2(cfg: SST2Config):
    raw_datasets = load_dataset(cfg.dataset_name, cfg.subset_name)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    def tokenize_function(example):
        return tokenizer(
            example["sentence"],
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets.set_format("torch")
    return tokenized_datasets, tokenizer


# --------------------------------------------------------------------------- #
# 3. Training utilities
# --------------------------------------------------------------------------- #
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": (preds == labels).mean()}


def train(cfg: SST2Config):
    accelerator = Accelerator()
    device = accelerator.device

    # Load data
    datasets, tokenizer = load_sst2(cfg)
    train_dataloader = torch.utils.data.DataLoader(
        datasets["train"], shuffle=True, batch_size=cfg.batch_size
    )
    eval_dataloader = torch.utils.data.DataLoader(
        datasets["validation"], batch_size=cfg.batch_size
    )

    # Load model and replace linear layers with adapters
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    replace_with_apt_adapter(model, cfg.rank_init, cfg.scaling)

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Scheduler
    num_training_steps = cfg.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Wrap everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Teacher copy for self‑distillation (initially same as student)
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    head_mask = torch.ones(
        model.bert.encoder.layer[0].attention.self.query.weight.size(0) // 12,
        device=device,
    )  # 12 heads for BERT‑base

    best_accuracy = 0.0
    metrics_log = []

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            # Cross‑entropy loss
            ce_loss = F.cross_entropy(logits, labels)

            # Self‑distillation loss (MSE between student and teacher logits)
            with torch.no_grad():
                teacher_logits = teacher_model(**batch).logits
            distill_loss = F.mse_loss(logits, teacher_logits)

            # Combined loss
            loss = ce_loss + 0.5 * distill_loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{cfg.num_epochs} | Loss: {avg_loss:.4f}")

        # ----- Evaluation -----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        accuracy = correct / total
        print(f"Validation accuracy: {accuracy:.4f}")
        metrics_log.append({"epoch": epoch + 1, "validation_accuracy": accuracy})

        # ----- Update teacher -----
        teacher_model.load_state_dict(model.state_dict())

        # ----- Prune low‑salience heads -----
        # Compute salience for each head
        grad_cache = {}
        for name, module in model.named_modules():
            if isinstance(module, APTAdapter):
                if module.weight.grad is not None:
                    grad_cache[name] = module.weight.grad.detach()
        salience = get_attention_head_salience(model, head_mask, grad_cache)
        # Determine heads to prune
        num_to_prune = int(cfg.prune_fraction * head_mask.size(0))
        if num_to_prune > 0:
            prune_indices = torch.argsort(salience)[:num_to_prune]
            head_mask[prune_indices] = 0
            print(f"Pruned heads: {prune_indices.tolist()}")

        # ----- Expand rank -----
        if (epoch + 1) % cfg.rank_increase_per_epoch == 0:
            new_rank = min(cfg.rank_max, cfg.rank_init + (epoch + 1) * cfg.rank_increase_per_epoch)
            for module in model.modules():
                if isinstance(module, APTAdapter):
                    module.expand_rank(new_rank)
            print(f"Expanded LoRA rank to {new_rank}")

    # Final evaluation on test set
    test_dataloader = torch.utils.data.DataLoader(
        datasets["test"], batch_size=cfg.batch_size
    )
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save final metrics
    output = {
        "test_accuracy": test_accuracy,
        "validation_metrics": metrics_log,
        "head_mask": head_mask.tolist(),
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results written to results.json")

if __name__ == "__main__":
    cfg = SST2Config()
    train(cfg)