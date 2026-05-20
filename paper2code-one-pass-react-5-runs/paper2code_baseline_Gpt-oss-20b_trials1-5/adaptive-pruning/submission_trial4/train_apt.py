#!/usr/bin/env python
"""
Minimal reproduction of the APT concept:
  • Load DistilBERT + LoRA adapter (PEFT)
  • Train on SST-2 for 1 epoch
  • Prune a fraction of attention heads
  • Continue training for 1 more epoch
  • Evaluate and write accuracy to output.json
"""

import json
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model

# --------------------------------------------------------------------------- #
# 1. Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------------------------------- #
# 2. Helper functions
# --------------------------------------------------------------------------- #
def compute_metrics(eval_pred):
    """Compute accuracy for the SST‑2 task."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": round(float(accuracy), 4)}

def prune_heads(model, prune_ratio=0.3):
    """
    Very simple head‑level pruning:
    For each layer, compute importance of each head as the sum of absolute
    query weights belonging to that head, keep the top (1‑prune_ratio) heads,
    and zero‑out the remaining heads' query/key/value weights.
    """
    for name, module in model.named_modules():
        # DistilBERT uses a custom SelfAttention module
        # We target the query/key/value linear layers
        if hasattr(module, "query") and hasattr(module, "key") and hasattr(module, "value"):
            # number of heads and head dimension
            num_heads = module.num_attention_heads
            head_dim = module.attention_head_size
            # Compute importance per head
            head_importance = []
            for h in range(num_heads):
                start = h * head_dim
                end = (h + 1) * head_dim
                # Query weights slice
                w_slice = module.query.weight.data[start:end]
                importance = w_slice.abs().sum().item()
                head_importance.append((importance, h))
            # Sort by importance descending
            head_importance.sort(key=lambda x: x[0], reverse=True)
            # Determine how many heads to keep
            keep = max(1, int(num_heads * (1 - prune_ratio)))
            heads_to_keep = [h for _, h in head_importance[:keep]]
            heads_to_remove = [h for _, h in head_importance[keep:]]
            if heads_to_remove:
                with torch.no_grad():
                    for h in heads_to_remove:
                        start = h * head_dim
                        end = (h + 1) * head_dim
                        module.query.weight.data[start:end] = 0
                        module.key.weight.data[start:end] = 0
                        module.value.weight.data[start:end] = 0
                print(f"Pruned {len(heads_to_remove)} heads in {name}")
            else:
                print(f"No heads pruned in {name}")
    print("Head pruning finished.")

# --------------------------------------------------------------------------- #
# 3. Main training routine
# --------------------------------------------------------------------------- #
def main():
    # Load data
    print("Loading SST‑2 dataset...")
    dataset = load_dataset("glue", "sst2")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Preprocess
    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding=False,
        )

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.remove_columns(["idx", "sentence"])
    encoded_dataset.set_format(type="torch")

    # Model + LoRA adapter
    print("Loading DistilBERT and adding LoRA adapter...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base_model, lora_config)

    # Training arguments – first epoch
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, padding=True),
        compute_metrics=compute_metrics,
    )

    # Train first epoch
    print("Training for 1 epoch...")
    trainer.train()

    # Prune heads
    print("Pruning attention heads (30% of heads)...")
    prune_heads(model, prune_ratio=0.3)

    # Continue training – second epoch
    print("Training for a second epoch...")
    training_args.num_train_epochs = 2
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, padding=True),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Evaluation
    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    accuracy = eval_results["eval_accuracy"]
    print(f"Accuracy: {accuracy}")

    # Save results
    with open("output.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)
    print("Results written to output.json")

if __name__ == "__main__":
    main()