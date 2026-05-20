import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskNet(nn.Module):
    """
    Simple feed‑forward network that outputs the probability of masking (i.e., 1)
    for each state.  The output is a single sigmoid value per state.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            Shape (batch, state_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) – probability that the step should be masked.
        """
        return self.net(state)