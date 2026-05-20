import torch
import torch.nn as nn

class MaskGenerator(nn.Module):
    """
    Lightweight 5‑layer CNN that outputs a 3‑channel mask.
    Architecture:
        Conv3x3  -> ReLU
        Conv3x3  -> ReLU
        Conv3x3  -> ReLU
        MaxPool2x2
        Conv3x3  -> ReLU
        Conv3x3  -> 1x1 -> 3 channels
    Input: image of size (B, 3, H, W) (already resized)
    Output: mask of same spatial size (B, 3, H, W)
    """
    def __init__(self, in_channels=3, out_channels=3, num_features=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, kernel_size=1),
            nn.Sigmoid()  # keep mask values in [0,1]
        )
        # Upsample factor 2 (because of MaxPool)
        self.scale_factor = 2

    def forward(self, x):
        mask_small = self.features(x)  # shape (B, 3, H/2, W/2)
        mask = patchwise_interpolate(mask_small, self.scale_factor)
        return mask