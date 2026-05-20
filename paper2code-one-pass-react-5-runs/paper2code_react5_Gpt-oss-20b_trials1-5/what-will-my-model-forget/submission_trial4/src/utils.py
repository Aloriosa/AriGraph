# src/utils.py
import random
import numpy as np
import torch
from torch import nn
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from datasets import load_dataset
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                         Utility helpers                                    #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Return tokenizer and model on the requested device."""
    if "bart" in model_name.lower():
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    elif "t5" in model_name.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return tokenizer, model


def encode_example(example, tokenizer, max_length=512):
    """
    Tokenise a single example.
    example: dict with keys 'input' and 'output'.
    Returns a dict suitable for the model.
    """
    inputs = tokenizer(
        example["input"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["output"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    inputs["labels"] = labels["input_ids"]
    return inputs


def train_one_epoch(model, dataloader, optimizer, device, grad_accum=1, num_steps=None):
    """
    Train for a single epoch or for a fixed number of steps.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    step = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum
        step += 1
        if num_steps is not None and step >= num_steps:
            break
    return total_loss / step if step else 0.0


def evaluate_example(model, tokenizer, example, device):
    """
    Generate a single prediction and compare to the gold output.
    Returns (prediction_str, target_str, is_correct)
    """
    model.eval()
    inputs = encode_example(example, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )
    pred_str = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    target_str = example["output"].strip()
    is_correct = pred_str == target_str
    return pred_str, target_str, is_correct


def compute_em(preds, targets):
    """Exact Match (EM) score: fraction of exact matches."""
    correct = sum(p.strip() == t.strip() for p, t in zip(preds, targets))
    return correct / len(preds) if preds else 0.0


# --------------------------------------------------------------------------- #
#                         Encoder MLP (for representation-based methods)    #
# --------------------------------------------------------------------------- #
class EncoderMLP(nn.Module):
    """
    Encodes an input–output pair into a low‑dimensional vector.
    Uses mean‑pooled encoder hidden states of input and output,
    concatenated, then passes through a small MLP.
    """

    def __init__(self, hidden_dim: int, out_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, input_emb, target_emb):
        """
        Forward pass for pre‑computed embeddings.
        """
        cat = torch.cat([input_emb, target_emb], dim=-1)
        return self.mlp(cat)


# --------------------------------------------------------------------------- #
#                         Representation utilities                           #
# --------------------------------------------------------------------------- #
def get_rep_vector(model, tokenizer, example, encoder, device):
    """
    Return a low‑dimensional representation vector for an example.
    """
    model.eval()
    with torch.no_grad():
        # encode input
        inp = tokenizer(
            example["input"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(device)
        inp_emb = encoder.encoder(inp["input_ids"], attention_mask=inp["attention_mask"])[0].mean(dim=1)

        # encode target
        tgt = tokenizer(
            example["output"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(device)
        tgt_emb = encoder.encoder(tgt["input_ids"], attention_mask=tgt["attention_mask"])[0].mean(dim=1)

        # concatenate and project
        rep = encoder.mlp(torch.cat([inp_emb, tgt_emb], dim=-1))
    return rep.squeeze(0).cpu()


def get_logit_target(model, tokenizer, example, device):
    """
    Returns the logits for the first non‑pad target token before/after update.
    """
    model.eval()
    with torch.no_grad():
        inputs = encode_example(example, tokenizer).to(device)
        outputs = model(**inputs)
        logits = outputs.logits  # [batch, seq_len, vocab]
        labels = inputs["labels"]
        target_mask = labels != tokenizer.pad_token_id
        # Find first non‑pad token index
        if target_mask.sum() == 0:
            # no valid target tokens; return zero logits
            return torch.zeros(logits.size(-1), device=device)
        first_idx = target_mask.nonzero(as_tuple=True)[0][0]
        logit_vec = logits[0, first_idx]
    return logit_vec.cpu()


# --------------------------------------------------------------------------- #
#                         Dataset helpers                                    #
# --------------------------------------------------------------------------- #
def load_p3_subset(num_examples=200, seed=42):
    """
    Load a small subset of the P3 training data.
    Each example is converted to {'input': prompt, 'output': answer, 'task_name': str}.
    """
    ds = load_dataset("p3", split="train")
    ds = ds.shuffle(seed=seed).select(range(num_examples))
    examples = []
    for idx, ex in enumerate(ds):
        output = ex["output"]
        if isinstance(output, list):
            output = output[0] if output else ""
        examples.append(
            {
                "input": ex["prompt"],
                "output": output,
                "task_name": ex["task_name"] if "task_name" in ex else f"task_{idx}",
            }
        )
    return examples


def load_mmlu_subset(num_examples=10, seed=42):
    """
    Load a small subset of the MMLU validation data.
    Each example is converted to {'input': question + choices, 'output': answer letter, 'task_name': str}.
    """
    ds = load_dataset("mmlu", split="validation")
    ds = ds.shuffle(seed=seed).select(range(num_examples))
    examples = []
    for idx, ex in enumerate(ds):
        question = ex["question"]
        choices = ex["choices"]
        choice_str = " ".join([f"{k}. {v}" for k, v in zip(choices["label"], choices["text"])])
        inp = f"Question: {question} Choices: {choice_str}"
        out = ex["answerKey"]  # e.g., 'A', 'B', ...
        examples.append(
            {
                "input": inp,
                "output": out,
                "task_name": ex["task_name"] if "task_name" in ex else f"mmlu_task_{idx}",
            }
        )
    return examples