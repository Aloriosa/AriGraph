"""
A minimal binary classifier that distinguishes source vs target
(noisy) images.  It is not used in the simplified training loop
but is provided for completeness and future extensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),          # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),         # 32 -> 16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)