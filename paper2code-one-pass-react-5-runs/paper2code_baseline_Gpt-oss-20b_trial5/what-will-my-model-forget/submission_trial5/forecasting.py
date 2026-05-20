#!/usr/bin/env python
"""
Utility functions for the forecasting model.
"""
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class RepresentationForecaster(nn.Module):
    """Same as in run_experiment.py – kept for potential reuse."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pooled_rep: torch.Tensor):
        return self.encoder(pooled_rep)

    def predict(self, repr_i, repr_j):
        logits = torch.matmul(repr_i, repr_j.T)
        probs = torch.sigmoid(logits)
        return probs.squeeze()