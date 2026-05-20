"""
Energy‑based adapter model.

The adapter receives the pooled representation of a candidate
output (sequence) and produces a scalar score g_θ(x).

Instead of training a tiny MLP on top of a frozen backbone
(what the toy version did), the full backbone is trainable.
This yields a model with 0.1B–0.3B parameters, matching the
paper’s description.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class EnergyAdapter(nn.Module):
    """
    Adapter that maps a pooled BERT/DeBERTa representation to a single scalar.
    The backbone is fully trainable (no freezing).
    """

    def __init__(self, model_name: str):
        """
        Args:
            model_name: huggingface model identifier (e.g. "microsoft/deberta-v3-base" or
                        "microsoft/deberta-v3-large")
        """
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=False,
            # We want the pooled [CLS] token, so no special pooling needed
        )
        hidden_dim = self.backbone.config.hidden_size

        # Simple linear head to scalar energy
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, input_ids, attention_mask):
        """
        Returns a scalar energy for each example in the batch.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # Use [CLS] token embedding (first token)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        energy = self.head(cls_emb)  # (batch, 1)
        return energy.squeeze(-1)  # (batch,)