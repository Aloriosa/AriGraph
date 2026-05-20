import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class Adapter(nn.Module):
    """
    Lightweight adapter: BERT encoder → single scalar score.
    Only the linear layer is trained; the encoder is frozen.
    """
    def __init__(self, encoder_name: str = "bert-base-uncased"):
        super().__init__()
        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        for p in self.encoder.parameters():
            p.requires_grad = False  # freeze encoder
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, text: list[str], device: torch.device):
        """
        Compute a scalar score for each text in the batch.
        """
        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        outputs = self.encoder(**encodings)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        scores = self.scorer(cls_emb).squeeze(-1)  # shape (batch,)
        return scores