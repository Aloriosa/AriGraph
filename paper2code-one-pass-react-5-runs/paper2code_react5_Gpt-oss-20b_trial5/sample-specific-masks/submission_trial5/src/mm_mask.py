import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerator(nn.Module):
    """
    Lightweight CNN that outputs a low‑resolution 3‑channel mask for every image.
    The mask is later up‑sampled by patch‑wise interpolation (nearest neighbour)
    to match the size of the shared noise pattern (224×224).
    Architecture:
        Conv(3→32)  → ReLU
        Conv(32→64, stride=2) → ReLU  # 112×112
        Conv(64→128, stride=2) → ReLU # 56×56
        Conv(128→256, stride=2) → ReLU # 28×28
        Conv(256→256) → ReLU          # 28×28
        Conv(256→3)   → Sigmoid
    """
    def __init__(self, in_channels=3, out_channels=3, feature_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim*2, 3, stride=2, padding=1)  # 112
        self.conv3 = nn.Conv2d(feature_dim*2, feature_dim*4, 3, stride=2, padding=1)  # 56
        self.conv4 = nn.Conv2d(feature_dim*4, feature_dim*8, 3, stride=2, padding=1)  # 28
        self.conv5 = nn.Conv2d(feature_dim*8, feature_dim*8, 3, padding=1)  # 28
        self.conv6 = nn.Conv2d(feature_dim*8, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Input: [B, 3, 224, 224]
        Output: [B, 3, 28, 28]  (values in [0,1])
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.sigmoid(self.conv6(x))
        return x