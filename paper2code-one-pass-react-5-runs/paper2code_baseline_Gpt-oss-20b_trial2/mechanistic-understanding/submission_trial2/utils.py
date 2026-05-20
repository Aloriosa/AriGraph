import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from unitary import ToxicityClassifier
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# 1. Dataset helpers
# --------------------------------------------------------------------------- #
def build_pairwise_dataset(
    tokenizer: AutoTokenizer,
    max_prompt_len: int = 128,
    max_gen_len: int = 20,
    num_samples: int = 200,
    seed: int = 42,
) -> List[dict]:
    """
    Build a synthetic pairwise dataset from Wikitext‑2.
    Each example contains:
        - prompt (first `max_prompt_len` tokens)
        - chosen (non‑toxic continuation)
        - rejected (toxic continuation)
    """
    torch.manual_seed(seed)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    examples = []
    for row in tqdm(dataset, desc="Building dataset"):
        text = row["text"].strip()
        if not text:
            continue
        # Tokenize prompt
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)[:max_prompt_len]
        if not prompt_ids:
            continue
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        # Generate chosen (non‑toxic) with greedy decoding
        chosen = generate_text(
            tokenizer, prompt, max_new_tokens=max_gen_len, do_sample=False
        )

        # Generate rejected (toxic) with a simple rule‑based toxic suffix
        # In practice you might use PPLM or a toxicity classifier to generate a toxic sample.
        # Here we simply append a known toxic phrase to guarantee toxicity.
        toxic_phrase = " I hate you"
        rejected = chosen + toxic_phrase

        examples.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return examples


def generate_text(
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """
    Generate text using the tokenizer and a *pre‑loaded* model.
    This helper is used only during dataset construction.
    """
    # Load a lightweight GPT‑2‑small for generation
    # (We do this lazily to avoid reloading for every example)
    global _generation_model
    if "_generation_model" not in globals():
        _generation_model = AutoModelForCausalLM.from_pretrained(
            "gpt2-medium",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        _generation_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(_generation_model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        output = _generation_model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# 2. Evaluation helpers
# --------------------------------------------------------------------------- #
def compute_toxicity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 20,
    batch_size: int = 8,
) -> float:
    """
    Compute average toxicity score over generated continuations.
    Uses the `unitary/toxic-bert` classifier.
    """
    classifier = ToxicityClassifier()
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        # Generate continuations
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        continuations = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        probs = classifier.predict(continuations)  # list of float in [0,1]
        results.extend(probs)
    return sum(results) / len(results) if results else 0.0


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    block_size: int = 128,
    batch_size: int = 8,
) -> float:
    """
    Compute perplexity on a HuggingFace dataset.
    """
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    # Concatenate all texts
    text = "\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=block_size)
    inputs = encodings["input_ids"].to(model.device)
    labels = inputs.clone()
    with torch.no_grad():
        outputs = model(inputs, labels=labels)
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity


def compute_f1(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_new_tokens: int = 20,
    batch_size: int = 8,
):
    """
    Compute token‑level F1 score on a reference continuation.
    For simplicity, we use the first `max_new_tokens` tokens of the *ground truth* as reference.
    """
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    references = []
    prompts = []
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        # Use the first `max_new_tokens` tokens as reference
        ref_ids = tokenizer.encode(text, add_special_tokens=False)[:max_new_tokens]
        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
        # The prompt is everything before the reference
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)[max_new_tokens:]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        references.append(ref_text.split())
        prompts.append(prompt_text)

    # Generate continuations
    all_preds = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        preds = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        all_preds.extend([p.split() for p in preds])

    # Compute precision, recall, F1
    tp = sum((set(r) & set(p)).__len__() for r, p in zip(references, all_preds))
    fp = sum((set(p) - set(r)).__len__() for r, p in zip(references, all_preds))
    fn = sum((set(r) - set(p)).__len__() for r, p in zip(references, all_preds))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1