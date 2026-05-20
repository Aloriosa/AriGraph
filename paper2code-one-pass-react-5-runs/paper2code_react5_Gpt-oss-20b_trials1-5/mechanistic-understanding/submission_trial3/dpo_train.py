"""
Fine‑tune GPT‑2‑medium with Direct Preference Optimization (DPO)
on the pairwise dataset produced by prepare_pairs.py.
"""

import json
import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from trl import DPOTrainer, DPOConfig
from tqdm import tqdm

BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer & base model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
base_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
base_model.to(device)

# Load reference model (same as base, but frozen)
ref_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
ref_model.to(device)
for p in ref_model.parameters():
    p.requires_grad = False

# Load pairwise data
with open("pairs.jsonl", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
dataset = Dataset.from_list(data)

def preprocess(example):
    prompt = example["prompt"]
    pos = example["positive"]
    neg = example["negative"]

    # Encode positive
    pos_ids = tokenizer(
        prompt + " " + pos,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    # Encode negative
    neg_ids = tokenizer(
        prompt + " " + neg,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return {
        "input_ids": pos_ids["input_ids"][0],
        "attention_mask": pos_ids["attention_mask"][0],
        "positive_ids": pos_ids["input_ids"][0],
        "negative_ids": neg_ids["input_ids"][0],
    }

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

trainer = DPOTrainer(
    model=base_model,
    ref_model=ref_model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=dict(
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="no",
        output_dir="dpo_output",
        bf16=torch.cuda.is_bf16_available(),
    ),
    dpo_config=DPOConfig(beta=0.5, use_kld=False),  # match paper hyper‑params
)

trainer.train()
trainer.save_model("dpo_gpt2")
print("DPO fine‑tuned model saved to dpo_gpt2")