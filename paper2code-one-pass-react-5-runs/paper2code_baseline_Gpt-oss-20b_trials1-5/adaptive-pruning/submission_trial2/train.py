#!/usr/bin/env python3
"""
Simplified APT reproduction for DistilBERT + SST-2.

The script:
1. Loads DistilBERT and SST-2 from HuggingFace.
2. Fine‑tunes with a LoRA adapter (peft) for a few epochs.
3. Prunes a percentage of feed‑forward neurons by zeroing out weight rows.
4. Fine‑tunes the pruned model again.
5. Outputs validation accuracies and writes them to metrics.json.
"""

import os
import json
import random
import numpy as np
import torch
import argparse
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_dataset(tokenizer, dataset, split):
    """
    Tokenize the dataset and keep only the needed columns.
    """
    def tokenize(example):
        return tokenizer(example["sentence"], truncation=True)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["sentence"])
    tokenized.set_format("torch")
    return tokenized[split]


def prune_ffn_layers(model, prune_ratio=0.3):
    """
    Prune a given percentage of feed‑forward output neurons in every linear layer.
    The pruning is done by zeroing the weight rows and bias entries.
    LoRA adapters are ignored.
    """
    for name, module in model.named_modules():
        # Skip LoRA modules
        if "lora" in name:
            continue
        # We target linear layers that map from hidden dim to hidden dim
        if isinstance(module, torch.nn.Linear):
            # Skip if input and output dims differ (e.g., classifier head)
            if module.out_features != module.in_features:
                continue
            weight = module.weight.data
            # Compute L2 norm for each output neuron (row)
            norms = torch.norm(weight, dim=1)
            # Determine number of neurons to prune
            k = int(prune_ratio * weight.size(0))
            if k == 0:
                continue
            # Find indices of smallest norms
            _, idx = torch.topk(norms, k, largest=False)
            # Zero out the selected rows
            weight[idx, :] = 0.0
            if module.bias is not None:
                module.bias[idx] = 0.0


def train_and_evaluate(
    model,
    tokenizer,
    train_ds,
    val_ds,
    output_dir,
    epochs,
    batch_size=8,
    learning_rate=2e-5,
):
    """
    Train the model using HuggingFace Trainer and evaluate.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        logging_steps=50,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to=[],
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results


# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------
def main():
    # Parse arguments (allow overriding config.ini if desired)
    parser = argparse.ArgumentParser(description="APT simplified reproduction.")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--prune_ratio", type=float, default=0.3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--epochs_phase1", type=int, default=2)
    parser.add_argument("--epochs_phase2", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    set_seed(42)

    # Load tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    raw_datasets = load_dataset("glue", args.task)
    # Use a subset of the training set for speed
    train_ds = preprocess_dataset(tokenizer, raw_datasets, "train")
    val_ds = preprocess_dataset(tokenizer, raw_datasets, "validation")

    # ------------------------------------------------------------------
    # Phase 1 – LoRA fine‑tune
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # DistilBERT uses these names
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model_lora = get_peft_model(base_model, lora_config)

    print("\n=== Phase 1: LoRA Fine‑Tune ===")
    eval_phase1 = train_and_evaluate(
        model_lora,
        tokenizer,
        train_ds,
        val_ds,
        output_dir="./tmp_phase1",
        epochs=args.epochs_phase1,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    acc_phase1 = eval_phase1.get("eval_accuracy", 0.0)
    print(f"Validation accuracy after Phase 1: {acc_phase1:.4f}")

    # ------------------------------------------------------------------
    # Phase 2 – Prune and fine‑tune
    # ------------------------------------------------------------------
    print("\n=== Phase 2: Prune 30% & Fine‑Tune ===")
    # Clone the LoRA model to preserve the original weights
    model_pruned = model_lora.transformer.base_model
    prune_ffn_layers(model_pruned, prune_ratio=args.prune_ratio)

    # Wrap pruned base model with LoRA again (reuse LoRA weights)
    model_pruned_lora = get_peft_model(
        model_pruned, lora_config, adapter_name="lora"
    )
    # Copy LoRA weights from previous model
    model_pruned_lora.load_state_dict(
        {k: v for k, v in model_lora.state_dict().items() if "lora" in k}
    )

    eval_phase2 = train_and_evaluate(
        model_pruned_lora,
        tokenizer,
        train_ds,
        val_ds,
        output_dir="./tmp_phase2",
        epochs=args.epochs_phase2,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    acc_phase2 = eval_phase2.get("eval_accuracy", 0.0)
    print(f"Validation accuracy after Phase 2: {acc_phase2:.4f}")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "phase1_accuracy": acc_phase1,
        "phase2_accuracy": acc_phase2,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nMetrics written to metrics.json")


if __name__ == "__main__":
    main()