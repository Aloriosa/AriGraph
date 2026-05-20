"""
A tiny adapter that is added to every UNet block.  
It consists of a 1×1 convolution that shifts the feature maps,
allowing the pre‑trained weights to stay frozen while still
providing a small capacity for adaptation.
"""

import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)