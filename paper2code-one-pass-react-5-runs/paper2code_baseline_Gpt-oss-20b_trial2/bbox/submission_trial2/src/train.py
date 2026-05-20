#!/usr/bin/env python
import argparse
import json
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from dataset import SimpleQADataset
from adapter import Adapter

def generate_candidates(model, tokenizer, prompt, num_candidates=5, device='cpu'):
    """
    Generate `num_candidates` continuations for a given prompt.
    """
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    outputs = model.generate(
        input_ids,
        do_sample=True,
        max_new_tokens=50,
        top_k=50,
        temperature=0.7,
        num_return_sequences=num_candidates,
        pad_token_id=tokenizer.eos_token_id
    )
    # Decode and strip
    candidates = [tokenizer.decode(o, skip_special_tokens=True).strip()
                  for o in outputs]
    # Remove prompt repetition
    candidates = [c[len(prompt):].strip() if c.startswith(prompt) else c.strip()
                  for c in candidates]
    return candidates

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data
    train_dataset = SimpleQADataset(args.train_file)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=lambda x: x)

    # Black‑box LLM
    blackbox_tokenizer = AutoTokenizer.from_pretrained(args.blackbox_name)
    blackbox_model = AutoModelForCausalLM.from_pretrained(
        args.blackbox_name).to(device)
    blackbox_model.eval()

    # Adapter
    adapter = Adapter().to(device)
    optimizer = optim.AdamW(adapter.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            queries = [item['question'] for item in batch]
            answers = [item['answer'] for item in batch]
            # Generate candidates
            all_candidates = []
            target_indices = []
            for q, gt in zip(queries, answers):
                prompt = f"Question: {q}\nAnswer:"
                cand = generate_candidates(
                    blackbox_model, blackbox_tokenizer, prompt,
                    num_candidates=args.num_candidates, device=device)
                # Ensure ground truth is included
                if gt not in cand:
                    cand.append(gt)
                idx = cand.index(gt)
                all_candidates.append(cand)
                target_indices.append(idx)

            # Flatten candidates for scoring
            flat_texts = []
            for cand_list in all_candidates:
                flat_texts.extend([f"Question: {q}\nAnswer: {c}"
                                   for q, c in zip(queries, cand_list)])
            scores = adapter(flat_texts)  # shape (B*K,)
            scores = scores.view(len(queries), -1)  # (B, K)

            # Compute loss
            loss = criterion(scores, torch.tensor(target_indices, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir,
                                 f"checkpoint-epoch-{epoch}")
        torch.save(adapter.state_dict(), ckpt_path)

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training JSONL file.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size.")
    parser.add_argument("--num_candidates", type=int, default=5,
                        help="Number of candidates generated per query.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--blackbox_name", type=str,
                        default="distilgpt2",
                        help="HuggingFace model name for black‑box LLM.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)