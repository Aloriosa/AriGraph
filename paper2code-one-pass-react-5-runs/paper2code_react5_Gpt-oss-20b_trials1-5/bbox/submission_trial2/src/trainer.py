"""
Training script for the EnergyAdapter using ranking‑based NCE loss.
"""

import argparse
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.adapter import EnergyAdapter
from src.utils import (
    load_strategyqa,
    load_gsm8k,
    load_truthfulqa,
    load_scienceqa,
    get_generation_pipeline,
    sample_candidates,
    compute_accuracy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BBox‑Adapter")
    parser.add_argument("--dataset", type=str, default="strategyqa")
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument(
        "--adapter_model",
        type=str,
        default="microsoft/deberta-v3-base",  # 86M parameters
        help="huggingface model name for the adapter (base or large)",
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/adapter_strategyqa")
    parser.add_argument("--log_dir", type=str, default="logs")
    return parser.parse_args()


def dataset_loader(name: str, split: str):
    if name == "strategyqa":
        return load_strategyqa(split)
    elif name == "gsm8k":
        return load_gsm8k(split)
    elif name == "truthfulqa":
        return load_truthfulqa(split)
    elif name == "scienceqa":
        return load_scienceqa(split)
    else:
        raise ValueError(f"Unsupported dataset {name}")


def main():
    args = parse_args()

    torch.manual_seed(42)
    random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------
    # 1. Load dataset
    # ---------------------------------------
    train_ds = dataset_loader(args.dataset, "train")
    test_ds = dataset_loader(args.dataset, "validation")

    # Keep a subset of the training data for speed
    train_ds = train_ds.shuffle(seed=42).select(range(args.train_size))
    train_ds = train_ds.add_column("idx", list(range(len(train_ds))))

    # ---------------------------------------
    # 2. Tokenizer for adapter
    # ---------------------------------------
    adapter_tokenizer = AutoTokenizer.from_pretrained(args.adapter_model)
    adapter_tokenizer.pad_token = adapter_tokenizer.eos_token

    # ---------------------------------------
    # 3. Adapter model
    # ---------------------------------------
    adapter = EnergyAdapter(model_name=args.adapter_model).to(device)
    adapter.train()

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
    )

    # ---------------------------------------
    # 4. LLM generator for negative samples
    # ---------------------------------------
    device_llm = 0 if torch.cuda.is_available() else -1
    llm_gen = get_generation_pipeline(
        model_name="mistralai/Mixtral-8x7B-v0.1",
        device=device_llm,
        max_length=args.max_seq_len,
        temperature=1.0,
    )

    # ---------------------------------------
    # 5. Training loop
    # ---------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Buffer to store best predictions from the previous epoch
    prev_epoch_buffer = {}

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    for epoch in range(args.epochs):
        adapter.train()
        epoch_loss = 0.0
        for batch in train_loader:
            # Positive samples: ground‑truth answers
            pos_answer = batch["label"]  # list of strings

            # Tokenize positive answer
            pos_enc = adapter_tokenizer(
                pos_answer,
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            ).to(device)

            # Generate negative candidates from LLM
            neg_cands_per_example = []
            for q, idx in zip(batch["prompt"], batch["idx"]):
                # LLM‑generated negatives
                neg = sample_candidates(llm_gen, q, num_candidates=4, device=device_llm)
                # Add previous epoch's best prediction as an extra negative if available
                if idx in prev_epoch_buffer:
                    neg.append(prev_epoch_buffer[idx])
                neg_cands_per_example.append(neg)

            # Flatten negatives
            flat_neg_cands = [cand for negs in neg_cands_per_example for cand in negs]

            neg_enc = adapter_tokenizer(
                flat_neg_cands,
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt",
            ).to(device)

            # Forward pass
            pos_energy = adapter(
                pos_enc["input_ids"], pos_enc["attention_mask"]
            )  # (batch,)
            neg_energy = adapter(
                neg_enc["input_ids"], neg_enc["attention_mask"]
            )  # (total_neg,)

            # Ranking‑based NCE loss
            exp_pos = torch.exp(pos_energy)
            # Compute sum of exp(neg) for each example
            exp_neg_sums = []
            start = 0
            for negs in neg_cands_per_example:
                end = start + len(negs)
                exp_neg_sums.append(torch.exp(neg_energy[start:end]).sum())
                start = end
            exp_neg_sums = torch.stack(exp_neg_sums)

            logits = torch.log(exp_pos + exp_neg_sums) - torch.log(exp_pos)
            loss = -logits.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}")

        # ---------------------------------------
        # 6. Evaluation on test set
        # ---------------------------------------
        adapter.eval()
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                q = batch["prompt"]
                # Generate 5 candidates per question
                cand_list = [
                    sample_candidates(llm_gen, question, 5, device=device_llm)
                    for question in q
                ]
                # Flatten
                cand_flat = [c for cs in cand_list for c in cs]
                cand_enc = adapter_tokenizer(
                    cand_flat,
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_seq_len,
                    return_tensors="pt",
                ).to(device)
                energies = adapter(
                    cand_enc["input_ids"], cand_enc["attention_mask"]
                )  # (batch*5,)

                # Pick candidate with highest energy
                energies = energies.view(len(q), -1)
                best_idx = energies.argmax(dim=1)
                best_answers = [cand_list[i][idx] for i, idx in enumerate(best_idx)]
                preds.extend(best_answers)

        # Convert answers to Yes/No
        preds_bin = ["Yes" if "yes" in p.lower() else "No" for p in preds]
        labels = [l for l in test_ds["label"]]
        acc = compute_accuracy(preds_bin, labels)
        print(f"  Test accuracy: {acc*100:.1f}%")

        # ---------------------------------------
        # 7. Store best predictions for the training set
        # ---------------------------------------
        train_buffer = {}
        train_loader_eval = DataLoader(train_ds, batch_size=1)
        with torch.no_grad():
            for batch in train_loader_eval:
                q = batch["prompt"][0]
                idx = batch["idx"][0]
                cand_list = sample_candidates(llm_gen, q, 5, device=device_llm)
                cand_enc = adapter_tokenizer(
                    cand_list,
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_seq_len,
                    return_tensors="pt",
                ).to(device)
                energies = adapter(
                    cand_enc["input_ids"], cand_enc["attention_mask"]
                ).squeeze(-1)
                best_idx = energies.argmax().item()
                train_buffer[idx] = cand_list[best_idx]
        # Update the buffer for the next epoch
        prev_epoch_buffer = train_buffer

        if acc > best_acc:
            best_acc = acc
            torch.save(adapter.state_dict(), os.path.join(args.output_dir, "adapter.pt"))
            print(f"  New best model saved (acc={best_acc*100:.1f}%)")

    # Save final model
    torch.save(adapter.state_dict(), os.path.join(args.output_dir, "adapter_final.pt"))
    print("Training finished.")


if __name__ == "__main__":
    main()