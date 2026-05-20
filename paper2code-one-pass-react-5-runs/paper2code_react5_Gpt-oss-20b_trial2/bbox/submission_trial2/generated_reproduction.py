# src/utils.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import random
import math
from typing import List, Tuple


def get_tokenizer(model_name: str):
    """Convenience helper to load a tokenizer."""
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
    using a causal LM (llm_model).  The function returns the summed
    log‑probability of all candidate tokens.
    """
    # Concatenate prompt and candidate with a space to separate them
    full_text = prompt + " " + candidate

    # Tokenize full text
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)
    # Tokenize prompt alone to get the boundary index
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)

    # Candidate token ids (exclude the prompt tokens)
    candidate_ids = full_ids[prompt_len:]
    if len(candidate_ids) == 0:
        return float("-inf")

    # Prepare input tensor
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    with torch.no_grad():
        outputs = llm_model(**enc)
        logits = outputs.logits  # (1, seq_len, vocab)

    # Compute log‑probs for each token
    logprobs = torch.log_softmax(logits, dim=-1)
    # Gather log‑probs for the candidate tokens
    cand_logprobs = logprobs[0, torch.arange(len(candidate_ids)), candidate_ids]
    total_logprob = cand_logprobs.sum().item()
    return total_logprob


def load_strategyqa(split: str):
    """
    Returns a HuggingFace dataset split for StrategyQA.
    """
    ds = load_dataset("strategyqa", split=split)
    # The dataset has columns: ['id', 'question', 'answer', 'choices'].
    # For our binary classification we only need 'question' and 'answer'.
    return ds.map(
        lambda x, i: {"prompt": x["question"], "label": x["answer"], "idx": i},
        with_indices=True,
    )


def load_gsm8k(split: str):
    """
    Returns a HuggingFace dataset split for GSM‑8K.
    """
    ds = load_dataset("gsm8k", split=split)
    return ds.map(
        lambda x, i: {"prompt": x["question"], "label": x["answer"], "idx": i},
        with_indices=True,
    )


def load_truthfulqa(split: str):
    """
    Returns a HuggingFace dataset split for TruthfulQA.
    """
    ds = load_dataset("truthful_qa", split=split)
    return ds.map(
        lambda x, i: {"prompt": x["question"], "label": x["answer"], "idx": i},
        with_indices=True,
    )


def load_scienceqa(split: str):
    """
    Returns a HuggingFace dataset split for ScienceQA.
    """
    ds = load_dataset("scienceqa", split=split)
    return ds.map(
        lambda x, i: {"prompt": x["question"], "label": x["answer"], "idx": i},
        with_indices=True,
    )


def compute_accuracy(preds: List[str], labels: List[str]) -> float:
    """
    Computes accuracy as the proportion of exact matches (case‑insensitive).
    If there are no labels, returns 0.0.
    """
    if not labels:
        return 0.0
    correct = sum(p.lower() == l.lower() for p, l in zip(preds, labels))
    return correct / len(labels)