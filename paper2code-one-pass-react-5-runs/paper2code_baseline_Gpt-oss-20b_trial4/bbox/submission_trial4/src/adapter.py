import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class Adapter(nn.Module):
    """
    Lightweight adapter that scores a candidate answer given the question.
    It encodes the concatenated [question] + [answer] pair using a
    DistilBERT encoder and outputs a scalar score.
    """
    def __init__(self, encoder_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, question: torch.Tensor, answer: torch.Tensor, attention_mask: torch.Tensor):
        """
        question, answer: token ids (batch, seq_len)
        attention_mask: (batch, seq_len)
        """
        # Concatenate question and answer with a separator token
        sep_id = self.encoder.config.sep_token_id or self.encoder.config.pad_token_id
        seq = torch.cat([question, torch.tensor([sep_id] * question.size(0), device=question.device), answer], dim=1)
        mask = torch.cat([attention_mask, torch.zeros(question.size(0), 1, device=attention_mask.device, dtype=torch.long), attention_mask], dim=1)
        outputs = self.encoder(input_ids=seq, attention_mask=mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        score = self.scorer(pooled).squeeze(-1)
        return score