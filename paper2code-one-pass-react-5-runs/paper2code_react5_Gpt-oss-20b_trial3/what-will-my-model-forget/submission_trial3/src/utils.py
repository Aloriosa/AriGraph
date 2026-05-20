import json
import logging
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

logger = logging.getLogger(__name__)


def load_squad() -> Tuple[List[dict], List[dict]]:
    """
    Load SQuAD v1.1 train (D_PT) and dev (D_R) splits.
    Each example is a dict with keys: 'id', 'title', 'context', 'question', 'answers'.
    """
    squad = load_dataset("squad", split="train")
    squad_dev = load_dataset("squad", split="validation")
    return squad, squad_dev


def encode_examples(
    examples: List[dict],
    tokenizer,
    max_length: int = 128,
    return_tensors: str = "pt",
):
    """
    Tokenize a list of SQuAD examples for sequence-to-sequence generation.
    """
    inputs = [
        f"question: {ex['question']} context: {ex['context']}" for ex in examples
    ]
    targets = [ex["answers"]["text"][0] if ex["answers"]["text"] else "" for ex in examples]
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=return_tensors,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=return_tensors,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def evaluate_em(
    model: AutoModelForSeq2SeqLM,
    tokenizer,
    examples: List[dict],
    batch_size: int = 8,
    max_length: int = 128,
) -> float:
    """
    Compute Exact Match (EM) on the given examples.
    """
    model.eval()
    em_count = 0
    total = len(examples)
    for i in range(0, total, batch_size):
        batch = examples[i : i + batch_size]
        inputs = [
            f"question: {ex['question']} context: {ex['context']}" for ex in batch
        ]
        input_ids = tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).input_ids
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=max_length, num_beams=4)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for pred, ex in zip(preds, batch):
            gold = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
            if pred.strip() == gold.strip():
                em_count += 1
    return em_count / total * 100.0


def get_pretraining_examples(
    squad: List[dict], num_samples: int = 100, seed: int = 42
) -> List[dict]:
    """
    Randomly sample `num_samples` examples from SQuAD train to form D_PT.
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(squad), size=num_samples, replace=False)
    return [squad[i] for i in idxs]


def get_error_examples(
    squad_dev: List[dict], model, tokenizer, num_errors: int = 20, seed: int = 43
) -> List[dict]:
    """
    Evaluate the base model on squad_dev and pick `num_errors` examples
    where the model fails (EM == 0 for the example).
    """
    errors = []
    for ex in squad_dev:
        inp = f"question: {ex['question']} context: {ex['context']}"
        input_ids = tokenizer(inp, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids, max_length=128, num_beams=4, early_stopping=True
            )
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        gold = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        if pred.strip() != gold.strip():
            errors.append(ex)
        if len(errors) >= num_errors:
            break
    return errors


def save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)