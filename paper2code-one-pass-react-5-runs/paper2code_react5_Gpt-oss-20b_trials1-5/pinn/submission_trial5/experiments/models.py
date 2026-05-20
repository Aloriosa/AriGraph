"""
MLP model used for all PINNs.

Three hidden layers with tanh activations and Xavier normal initialization.
Bias terms are set to zero.
"""

import torch
import torch.nn as nn
from torch.nn import init
from experiments.config import ACTIVATION


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(3):
            linear = nn.Linear(in_dim, width, bias=True)
            # Xavier normal init
            init.xavier_normal_(linear.weight)
            init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(ACTIVATION)
            in_dim = width
        # Output layer
        out = nn.Linear(in_dim, output_dim, bias=True)
        init.xavier_normal_(out.weight)
        init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)