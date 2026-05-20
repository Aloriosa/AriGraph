#!/usr/bin/env python3
"""
Train a GPT‑2‑medium model with Direct Preference Optimization (DPO)
on a synthetic pairwise preference dataset.

Output:
    - Saved fine‑tuned model in ./dpo_model/
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from trl import DPOTrainer, DPOConfig
from accelerate import Accelerator

from utils import build_pairwise_dataset

# --------------------------- Configuration --------------------------- #
MODEL_NAME = "gpt2-medium"          # Base model
OUTPUT_DIR = Path("dpo_model")
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-6
BETAS = 0.1
MAX_PROMPT_LEN = 128
MAX_GEN_LEN = 20
NUM_SAMPLES = 200  # Size of synthetic dataset

# --------------------------- Main ----------------------------------- #
def main():
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    # Load tokenizer and model (FP16 + bnb for speed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # GPT‑2 has no pad

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 8‑bit quantization
        llm_int8_skip_modules=["lm_head"],  # keep head FP32
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()  # will be set to train mode by trainer

    # Build synthetic pairwise dataset
    print("Building synthetic pairwise dataset...")
    pairwise_examples = build_pairwise_dataset(
        tokenizer,
        max_prompt_len=MAX_PROMPT_LEN,
        max_gen_len=MAX_GEN_LEN,
        num_samples=NUM_SAMPLES,
    )
    ds = Dataset.from_list(pairwise_examples)

    # DPO config
    dpo_config = DPOConfig(
        beta=BETAS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_epochs=NUM_EPOCHS,
        optim="adamw_torch",
        scheduler="cosine",
        lr_warmup_steps=50,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # use same weights as reference
        tokenizer=tokenizer,
        train_dataset=ds,
        **dpo_config.to_dict(),
    )

    # Train
    trainer.train()

    # Save fine‑tuned model
    print(f"Saving fine‑tuned model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    trainer.tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()