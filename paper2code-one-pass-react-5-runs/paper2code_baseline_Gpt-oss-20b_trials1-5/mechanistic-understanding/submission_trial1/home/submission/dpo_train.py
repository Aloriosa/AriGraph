"""
Fine‑tune a causal LM with Direct Preference Optimization (DPO)
using the `trl` library.
"""

import torch
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def train_dpo(pairwise_list, tokenizer, device="cpu",
              epochs=1, batch_size=4, learning_rate=1e-6, beta=0.1):
    """
    pairwise_list: list of dicts with keys 'prompt', 'chosen', 'rejected'
    """
    # Convert to HuggingFace Dataset
    ds = Dataset.from_list(pairwise_list)

    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    model.to(device)

    config = DPOConfig(
        learning_rate=learning_rate,
        beta=beta,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        max_epochs=epochs,
        max_length=256,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # we use the base weights as reference
        tokenizer=tokenizer,
        train_dataset=ds,
        eval_dataset=None,
        config=config,
    )

    trainer.train()
    return trainer.model