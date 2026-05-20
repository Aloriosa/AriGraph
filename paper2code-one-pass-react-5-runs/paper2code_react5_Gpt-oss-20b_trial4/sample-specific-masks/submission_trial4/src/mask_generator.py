import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerator(nn.Module):
    """
    Lightweight 5‑layer CNN that outputs a 3‑channel mask for each image.
    The design follows the paper: 5 conv layers with 2 MaxPool layers.
    """
    def __init__(self):
        super().__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        # Layer 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        # Layer 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        # Layer 4
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(32)
        # Layer 5
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(3)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W)  ->  (B,3,H,W)  mask in [0,1]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)          # /2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)          # /4
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        # Patch‑wise interpolation: upsample by nearest to original size
        x = F.interpolate(x, scale_factor=4, mode='nearest')
        # Clamp to [0,1] and apply sigmoid for smooth masks
        return torch.sigmoid(x)