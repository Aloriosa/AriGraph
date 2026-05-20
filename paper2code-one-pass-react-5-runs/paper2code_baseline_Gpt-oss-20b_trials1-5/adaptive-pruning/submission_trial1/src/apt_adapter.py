import torch
import torch.nn as nn
from transformers import AdamW

class APTAdapter(nn.Module):
    """
    A minimal LoRA‑style adapter with dynamic rank.
    It injects a low‑rank correction in the hidden dimension.
    """
    def __init__(self, hidden_size, rank=8, scale=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.scale = scale
        # LoRA matrices: W_A -> (rank, hidden_size), W_B -> (hidden_size, rank)
        self.W_A = nn.Parameter(torch.randn(rank, hidden_size))
        self.W_B = nn.Parameter(torch.randn(hidden_size, rank))
        # Bias for the adapter (optional)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        # x: (batch, seq_len, hidden)
        # Adaptation: x + scale * (x @ W_A.T @ W_B.T)
        out = x + self.scale * (x @ self.W_A.T @ self.W_B.T) + self.bias
        return out

    def to(self, device):
        super().to(device)
        self.W_A.data = self.W_A.data.to(device)
        self.W_B.data = self.W_B.data.to(device)
        self.bias.data = self.bias.data.to(device)
        return self