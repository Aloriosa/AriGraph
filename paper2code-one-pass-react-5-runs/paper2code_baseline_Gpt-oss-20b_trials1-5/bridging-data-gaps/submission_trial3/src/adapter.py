"""
A tiny learnable adapter that adds a scalar shift per diffusion timestep.
The adapter is a vector of shape (T,) where T is the number of timesteps.
During training we multiply the UNet's predicted noise by (1 + adapter[t]).
"""

import torch
import torch.nn as nn

class TimeAdapter(nn.Module):
    def __init__(self, num_timesteps: int):
        super().__init__()
        # Initialize all adapters to zero (no shift)
        self.adapter = nn.Parameter(torch.zeros(num_timesteps))

    def forward(self, eps_pred: torch.Tensor, timesteps: torch.Tensor):
        """
        eps_pred: (B, C, H, W)
        timesteps: (B,) int tensor
        """
        # Gather the adapter for each sample in the batch
        shift = self.adapter[timesteps].view(-1, 1, 1, 1)
        return eps_pred * (1.0 + shift)