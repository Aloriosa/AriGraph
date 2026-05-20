# train_apt.py
"""
Training script for the lightweight APT implementation.

Usage example:
    python train_apt.py \
        --model_name bert-base-uncased \
        --task sst2 \
        --epochs 3 \
        --batch_size 32 \
        --lr 2e-5 \
        --output_dir ./results
"""

import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from apt_adapter import APTBertForSequenceClassification

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long)
    }

def evaluate(model, loader, device):
    model.eval()
    metric = load_metric('accuracy')
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            loss, logits = model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)
            preds = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=preds.cpu(), references=labels.cpu())
    return metric.compute()['accuracy']

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load data
    raw_datasets = load_dataset('glue', args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length')

    encoded_datasets = raw_datasets.map(preprocess_function, batched=True)
    train_dataset = encoded_datasets['train'].shuffle(seed=42).select(range(2000))  # small subset for quick demo
    val_dataset = encoded_datasets['validation']

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # 2. Load model
    config = AutoTokenizer.from_pretrained(args.model_name).config
    model = APTBertForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        init_rank=args.init_rank,
        max_rank=args.max_rank,
        rank_increase_epochs=args.rank_increase_epochs,
        target_head_sparsity=args.head_sparsity,
        device=device
    ).to(device)

    # 3. Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.05 * total_steps),
                                                num_training_steps=total_steps)

    # 4. Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            loss, logits = model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Rank adjustment at scheduled epochs
        if epoch in args.rank_increase_epochs:
            print(f"Epoch {epoch}: increasing LoRA rank")
            model.increase_rank()

        # Validation
        acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} – Validation accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            # Save best model
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))

    # 5. Final evaluation
    final_acc = evaluate(model, val_loader, device)
    print(f"Final validation accuracy: {final_acc:.4f}")

    # Save metrics
    results = {
        'final_accuracy': float(final_acc),
        'best_accuracy': float(best_acc),
        'model_path': os.path.join(args.output_dir, 'best_model.pt')
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APT training script")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Pretrained model name")
    parser.add_argument("--task", type=str, default="sst2",
                        help="GLUE task name (e.g., sst2, mnli)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to store outputs")
    parser.add_argument("--init_rank", type=int, default=8,
                        help="Initial LoRA rank")
    parser.add_argument("--max_rank", type=int, default=32,
                        help="Maximum LoRA rank")
    parser.add_argument("--rank_increase_epochs", nargs="+", type=int,
                        default=[1, 2],
                        help="Epochs at which to increase LoRA rank")
    parser.add_argument("--head_sparsity", type=float, default=0.4,
                        help="Target sparsity of attention heads (fraction to prune)")
    args = parser.parse_args()
    main(args)