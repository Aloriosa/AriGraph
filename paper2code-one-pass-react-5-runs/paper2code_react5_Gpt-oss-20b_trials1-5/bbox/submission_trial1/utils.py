# utils.py
import torch
from transformers import AutoTokenizer, pipeline
from datasets import Dataset
from tqdm import tqdm

def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def generate_candidates(
    tokenizer,
    prompt: str,
    model_name: str,
    num_candidates: int = 3,
    max_length: int = 256,
    temperature: float = 0.7,
    device: str = None,
):
    """
    Generate a list of candidate completions for a given prompt using a
    local language model.  The function uses HuggingFace's pipeline API.
    """
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    completions = pipe(prompt, num_return_sequences=num_candidates)
    return [c["generated_text"] for c in completions]

def prepare_batch(
    batch,
    tokenizer,
    max_length: int = 512,
):
    """
    Tokenize a list of texts in a batch.
    """
    texts = batch["text"]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc

def compute_nce_loss(scores, pos_mask):
    """
    Simplified ranking‑based NCE loss.
    `scores` – tensor of shape [batch_size] containing scores for each sample.
    `pos_mask` – boolean mask indicating which indices are positive.
    """
    # Positive scores
    pos_scores = scores[pos_mask]
    # Negative scores
    neg_scores = scores[~pos_mask]
    # Compute logits for all samples
    logits = scores
    # Softmax denominator
    denom = torch.logsumexp(logits, dim=0)
    # Numerator (positive)
    num = torch.logsumexp(pos_scores, dim=0)
    loss = -(num - denom)
    return loss

def load_gsm8k(split: str = "train[:10%]"):
    """
    Load a small subset of GSM8K for quick experiments.
    The dataset contains a 'question' and an 'answer' field.
    """
    ds = Dataset.from_dict({"question": [], "answer": []})
    ds = ds.add_column("question", [])
    ds = ds.add_column("answer", [])
    # Use the HuggingFace datasets library to load GSM8K
    from datasets import load_dataset

    raw = load_dataset("gsm8k", split=split)
    ds = raw.select_columns(["question", "answer"])
    return ds

def evaluate_adapter(adapter, tokenizer, test_ds, num_candidates=3):
    """
    Run inference on the test set and compute accuracy.
    """
    adapter.eval()
    correct = 0
    total = 0
    for example in tqdm(test_ds, desc="Evaluating"):
        prompt = example["question"]
        # Generate candidates
        candidates = generate_candidates(
            tokenizer,
            prompt,
            adapter.encoder.config._name_or_path,
            num_candidates=num_candidates,
        )
        # Score each candidate
        batch = {"text": candidates}
        enc = prepare_batch(batch, tokenizer)
        scores = adapter(enc["input_ids"].to(adapter.device), enc["attention_mask"].to(adapter.device))
        best_idx = torch.argmax(scores).item()
        pred = candidates[best_idx].strip()
        # The ground‑truth answer may have trailing spaces/newlines
        gt = example["answer"].strip()
        if pred == gt:
            correct += 1
        total += 1
    return correct / total