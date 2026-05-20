#!/usr/bin/env python
import os
import torch
import random
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
MODEL_NAME = "gpt2-medium"
OUTPUT_DIR = "./dpo_model"
MAX_LENGTH = 50
NUM_PAIRS = 200
BATCH_SIZE = 4
LR = 1e-6
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# --------------------------------------------------------------------------- #
# Load base model
# --------------------------------------------------------------------------- #
print(f"Loading base model {MODEL_NAME} ...")
# Enable 8-bit if GPU available to reduce memory
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True if torch.cuda.is_available() else False,
    llm_int8_enable_fp32_cpu_offload=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config
).to(DEVICE)

# --------------------------------------------------------------------------- #
# Build pairwise dataset
# --------------------------------------------------------------------------- #
print("Building pairwise dataset ...")
from src.data import build_pairwise_dataset

pairwise_ds = build_pairwise_dataset(
    tokenizer,
    base_model,
    num_pairs=NUM_PAIRS,
    max_length=MAX_LENGTH,
)

# --------------------------------------------------------------------------- #
# DPO Trainer
# --------------------------------------------------------------------------- #
print("Setting up DPO trainer ...")
trainer = DPOTrainer(
    model=base_model,
    ref_model=base_model,  # reference model is frozen
    tokenizer=tokenizer,
    args=dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="no",
        fp16=True,
    ),
    train_dataset=pairwise_ds,
    max_prompt_length=MAX_LENGTH,
    max_length=MAX_LENGTH + 20,  # to allow for continuation
    dpo_beta=0.1,
)

# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
print("Training ...")
trainer.train()

# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
print(f"Saving fine-tuned model to {OUTPUT_DIR} ...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training completed.")