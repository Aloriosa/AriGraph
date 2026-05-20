import random
from typing import List, Tuple, Dict
from datasets import Dataset
from transformers import AutoTokenizer

def generate_synthetic_examples(
    num_examples: int,
    min_len: int = 5,
    max_len: int = 15,
    vocab: List[str] = None,
) -> List[Dict]:
    """Generate synthetic input–output pairs."""
    if vocab is None:
        vocab = [f"word{i}" for i in range(1000)]
    examples = []
    for _ in range(num_examples):
        inp_len = random.randint(min_len, max_len)
        out_len = random.randint(min_len, max_len)
        inp = " ".join(random.choices(vocab, k=inp_len))
        out = " ".join(random.choices(vocab, k=out_len))
        examples.append({"input_text": inp, "target_text": out})
    return examples

def create_dataset(
    num_pretrain: int = 50,
    num_online: int = 5,
) -> Tuple[Dataset, Dataset]:
    """Return pre‑training and online data as HuggingFace datasets."""
    pretrain = generate_synthetic_examples(num_pretrain)
    online = generate_synthetic_examples(num_online)
    return Dataset.from_dict(pretrain), Dataset.from_dict(online)