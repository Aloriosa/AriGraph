import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class Adapter(nn.Module):
    """
    Small transformer encoder that scores a (question + answer) pair.
    Output is a single scalar energy value: lower = more likely.
    """
    def __init__(self, encoder_name='bert-base-cased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        # Linear layer maps [CLS] hidden state to a scalar
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, texts):
        """
        texts: List[str] of concatenated question and answer.
        Returns: Tensor of shape (batch_size,) with scalar scores.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(next(self.parameters()).device)

        outputs = self.encoder(**inputs)
        cls = outputs.last_hidden_state[:, 0]  # [CLS] token
        scores = self.scorer(cls).squeeze(-1)  # shape (batch_size,)
        return scores