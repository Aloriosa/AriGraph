import torch
import torch.nn as nn
from transformers import DistilBertModel

class Adapter(nn.Module):
    """
    Lightweight adapter that scores a concatenated query + answer pair.
    Uses DistilBERT as the backbone and a single linear head.
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name)
        self.head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls).squeeze(-1)  # shape (batch,)
        return logits