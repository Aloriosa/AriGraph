import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMaskGenerator(nn.Module):
    """Base class for mask generators."""
    def __init__(self, l: int):
        """
        Parameters
        ----------
        l : int
            Patch size exponent (2^l).  The generator will have l
            max‑pool layers, producing an output of size
            H//(2^l) × W//(2^l).
        """
        super().__init__()
        self.l = l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a mask of shape (B, 3, h', w') with values in [0,1].
        """
        raise NotImplementedError


class ResNetMaskGenerator(BaseMaskGenerator):
    """
    5‑layer CNN for ResNet backbones.

    Architecture:
        Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> ReLU
        MaxPool2x2 -> Conv3x3 -> ReLU -> Conv3x3 -> ReLU
        Conv3x3 -> ReLU -> Conv3x3 -> ReLU
        Conv3x3 -> Sigmoid (output 3 channels)
    """

    def __init__(self, l: int = 3):
        super().__init__(l)
        # 5 layers, 3 Conv3x3 + 2 MaxPool2x2
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # -> /2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.features(x)


class ViTMaskGenerator(BaseMaskGenerator):
    """
    6‑layer CNN for ViT backbones.

    Architecture similar to ResNetMaskGenerator but with an extra
    Conv3x3 layer to accommodate the larger input size.
    """

    def __init__(self, l: int = 3):
        super().__init__(l)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # /2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.features(x)