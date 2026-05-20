"""Utility functions for handling the base model and fine‑tuning."""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict, List

# Global device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_base_model(
    model_name: str = "t5-base", max_length: int = 128, device: torch.device = DEVICE
):
    """Load a T5‑base model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def fine_tune_one_example(
    model,
    tokenizer,
    example: Dict,
    K: int = 30,
    lr: float = 1e-4,
    max_length: int = 128,
):
    """
    Fine‑tune the model for K gradient steps on a single example.
    The example is a dict with keys 'text' (prompt) and 'label' (int class).
    We cast the class label into a string token via the tokenizer.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Build input and target
    input_text = f"Classify this news: {example['text']}"
    target_text = str(example["label"])

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    ).to(DEVICE)
    targets = tokenizer(
        target_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    ).to(DEVICE)

    # Prepare labels (ignore padding)
    labels = targets["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100

    for _ in range(K):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    return model  # the fine‑tuned model