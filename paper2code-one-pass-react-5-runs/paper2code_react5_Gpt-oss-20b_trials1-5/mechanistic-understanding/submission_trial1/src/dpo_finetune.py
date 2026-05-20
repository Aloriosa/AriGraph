#!/usr/bin/env python3
"""
Fine‑tune GPT‑2‑medium with Direct Preference Optimization (DPO)
using the trl library.
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pairwise_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pairwise data
    with open(args.pairwise_path) as f:
        pairwise = json.load(f)

    # Convert to HF Dataset in the format required by trl (prompt, chosen, rejected)
    dataset = Dataset.from_dict({
        "prompt": [ex["prompt"] for ex in pairwise],
        "chosen": [ex["chosen"] for ex in pairwise],
        "rejected": [ex["rejected"] for ex in pairwise]
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    # DPO configuration (matching the paper’s hyper‑parameters roughly)
    dpo_config = DPOConfig(
        learning_rate=1e-5,
        beta=0.1,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_grad_norm=10,
        logging_steps=50,
        save_steps=200,
        output_dir=args.output_dir,
        seed=42,
        fp16=True if torch.cuda.is_available() else False
    )

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=dpo_config
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"DPO fine‑tuned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()