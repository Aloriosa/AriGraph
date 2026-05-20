import torch
import torch.nn as nn

class ExpandingRouter(nn.Module):
    """
    Expanding linear soft‑max router that produces a weight vector
    over the adapters of one transformer layer.
    """
    def __init__(self, dim: int, num_adapters: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, num_adapters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CLS token representation, shape [batch, dim]
        Returns:
            weights: softmax over adapters, shape [batch, num_adapters]
        """
        logits = x @ self.weight  # [batch, num_adapters]
        return torch.softmax(logits, dim=-1)

    def expand(self, new_dim: int = 1):
        """
        Add a new column to the weight matrix when a new adapter is added.
        """
        with torch.no_grad():
            new_col = torch.randn(self.weight.size(0), new_dim)
            self.weight = nn.Parameter(torch.cat([self.weight, new_col], dim=1))