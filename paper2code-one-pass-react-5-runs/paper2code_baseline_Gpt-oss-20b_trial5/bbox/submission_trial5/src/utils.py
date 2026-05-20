import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# 1. Reproducibility helpers
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------
# 2. Dataset utilities
# ------------------------------------------------------------
class QADataset(Dataset):
    """Simple JSONL QA dataset."""
    def __init__(self, path: Path):
        self.samples = []
        for line in path.read_text().splitlines():
            if line.strip():
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.samples[idx]["question"], self.samples[idx]["answer"]

def collate_fn(batch):
    questions, answers = zip(*batch)
    return list(questions), list(answers)

# ------------------------------------------------------------
# 3. Helper: generate candidates with GPT‑2
# ------------------------------------------------------------
def generate_candidates(gpt_tokenizer, gpt_model, question: str,
                        n_candidates: int = 5,
                        max_length: int = 20,
                        device: torch.device = torch.device("cpu")) -> List[str]:
    """
    Generate `n_candidates` answers for a question using GPT‑2 beam search.
    """
    input_ids = gpt_tokenizer.encode(question, return_tensors="pt").to(device)
    # GPT‑2 generation with beam search
    outputs = gpt_model.generate(
        input_ids,
        max_new_tokens=max_length,
        num_beams=n_candidates,
        early_stopping=True,
        no_repeat_ngram_size=2,
        do_sample=False,
    )
    return [gpt_tokenizer.decode(out, skip_special_tokens=True).strip()
            for out in outputs]

# ------------------------------------------------------------
# 4. Helper: compute exact‑match accuracy
# ------------------------------------------------------------
def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())

def write_jsonl(path: Path, data: List[dict]):
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")