import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    """
    Very small energy-based adapter.
    It takes the embedding of the concatenated question+answer and produces a scalar score.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, hidden_dim)
        returns: (batch, 1) logits
        """
        return self.linear(x)