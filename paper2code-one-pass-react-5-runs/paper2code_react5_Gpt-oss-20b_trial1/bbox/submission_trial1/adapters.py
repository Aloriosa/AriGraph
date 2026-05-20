# adapters.py
import torch
import torch.nn as nn
from transformers import AutoModel

class Adapter(nn.Module):
    """
    Lightweight adapter that scores candidate responses.
    It uses a frozen pre‑trained encoder (e.g., DeBERTa) and a single
    linear output layer to produce a scalar score.
    """
    def __init__(self, encoder_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, return_dict=True)
        # Freeze encoder parameters – we only train the final linear head
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Returns a score for each example in the batch.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        score = self.scorer(pooled).squeeze(-1)
        return score