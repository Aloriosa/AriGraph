import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import random
import math
from typing import List, Tuple


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def get_generation_pipeline(
    model_name: str,
    device: int = -1,
    max_length: int = 512,
    temperature: float = 1.0,
):
    """
    Returns a HuggingFace text‑generation pipeline.
    """
    return pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        device=device,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True,
    )


def sample_candidates(
    generator,
    prompt: str,
    num_candidates: int,
    device: int = -1,
) -> List[str]:
    """
    Generates `num_candidates` completions for the given prompt.
    """
    outputs = generator(
        prompt,
        num_return_sequences=num_candidates,
        return_full_text=False,
    )
    # Each output is a dict with 'generated_text'
    return [o["generated_text"].strip() for o in outputs]


def compute_logprob_llm(
    llm_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    candidate: str,
    device: torch.device,
) -> float:
    """
    Computes the log‑probability of `candidate` conditioned on `prompt`
    using a causal LM (llm_model).
    """
    enc = tokenizer(
        [prompt, candidate],
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        outputs = llm_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab)

    # Shift logits by one to align with next token prediction
    logprobs = torch.log_softmax(logits, dim=-1)
    logprobs = logprobs[:, :-1, :]  # remove last position

    # Determine token ids of candidate
    prompt_len = attention_mask[0].sum().item()
    candidate_ids = input_ids[0, prompt_len:]

    if len(candidate_ids) == 0:
        return float("-inf")

    # Gather log probabilities for each candidate token
    candidate_logprobs = logprobs[0, torch.arange(len(candidate_ids)), candidate_ids]
    total_logprob = candidate_logprobs.sum().item()
    return total_logprob


def load_strategyqa(split: str):
    """
    Returns a HuggingFace dataset split for StrategyQA.
    """
    ds = load_dataset("strategyqa", split=split)
    # The dataset has columns: ['id', 'question', 'answer', 'choices'].
    # For our binary classification we only need 'question' and 'answer'.
    # Add an index column to track examples across epochs
    return ds.map(
        lambda x, i: {"prompt": x["question"], "label": x["answer"], "idx": i},
        with_indices=True,
    )


def load_gsm8k(split: str):
    """
    Returns a HuggingFace dataset split for GSM‑8K.
    """
    ds = load_dataset("gsm8k", split=split)
    # Keep the original 'question' and 'answer' columns
    return ds.map(
        lambda x, i: {"prompt": x["question"], "label": x["answer"], "idx": i},
        with_indices=True,
    )


def load_truthfulqa(split: str):
    """
    Returns a HuggingFace dataset split for TruthfulQA.
    """
    ds = load_dataset("truthful_qa", split=split)
    # The dataset contains 'question' and 'answer' (Yes/No)
    return ds.map(
        lambda x, i: {"prompt": x["question"], "label": x["answer"], "idx": i},
        with_indices=True,
    )


def load_scienceqa(split: str):
    """
    Returns a HuggingFace dataset split for ScienceQA.
    """
    ds = load_dataset("scienceqa", split=split)
    # Keep 'question' and 'answer' (string)
    return ds.map(
        lambda x, i: {"prompt": x["question"], "label": x["answer"], "idx": i},
        with_indices=True,
    )


def compute_accuracy(preds, labels):
    """
    Computes accuracy as the proportion of exact matches.
    Comparison is case‑insensitive.
    """
    if not labels:
        return 0.0
    correct = sum(p.lower() == l.lower() for p, l in zip(preds, labels))
    return correct / len(labels)