import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    Lightweight adapter used in SEMA. It implements a bottleneck
    architecture : down-projection → ReLU → up-projection.
    """
    def __init__(self, dim: int, bottleneck: int = 64):
        """
        Args:
            dim: dimension of the hidden state (ViT hidden size)
            bottleneck: size of the bottleneck projection
        """
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.
        """
        return self.up(self.act(self.down(x)))