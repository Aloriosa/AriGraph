"""
Sentence embedding utilities using HuggingFace transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

from .config import DEVICE


class SentenceEmbedder:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.to(DEVICE)
        self.model.eval()

    def embed(self, texts: List[str]) -> torch.Tensor:
        """Return mean‑pooled embeddings for a list of texts."""
        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(DEVICE)
            outputs = self.model(**encoded)
            token_embeddings = outputs.last_hidden_state  # (batch, seq, dim)
            # mean pooling
            mask = encoded.attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts
            return mean_pooled.cpu()