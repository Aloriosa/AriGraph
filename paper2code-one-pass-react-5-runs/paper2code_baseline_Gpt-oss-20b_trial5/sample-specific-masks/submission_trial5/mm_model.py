import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerator(nn.Module):
    """
    5‑layer CNN that outputs a 3‑channel mask.
    All conv layers use 3×3 kernels, stride 1, padding 1.
    After each conv (except the last), a ReLU and a 2×2 max‑pool is applied.
    """
    def __init__(self, in_channels=3, out_channels=3, hidden_dim=64, num_layers=5):
        super().__init__()
        layers = []
        # Input conv
        layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        # Hidden convs
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
        # Output conv
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1))
        # No activation; mask values are learned in [-1, 1] range
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, H, W) resized image
        mask = self.net(x)  # (B, 3, H', W')
        return mask

class SMMWrapper(nn.Module):
    """
    Wrapper that combines:
    * a frozen backbone (ResNet or ViT)
    * a shared learnable noise pattern δ
    * a sample‑specific mask generator
    """
    def __init__(self, backbone, input_size):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()  # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Shared noise pattern δ, initialized to zeros
        self.delta = nn.Parameter(torch.zeros(3, input_size, input_size))

        # Mask generator
        self.mask_gen = MaskGenerator(in_channels=3,
                                     out_channels=3,
                                     hidden_dim=64,
                                     num_layers=5)

    def forward(self, x):
        """
        x: original image batch (B, 3, H, W)
        """
        # Resize input to backbone input size
        x_resized = F.interpolate(x, size=(self.delta.shape[1], self.delta.shape[2]),
                                  mode='bilinear', align_corners=False)

        # Generate mask
        mask = self.mask_gen(x_resized)  # (B, 3, H', W')
        # Upsample mask to match delta shape
        mask_up = F.interpolate(mask, size=(self.delta.shape[1], self.delta.shape[2]),
                                mode='nearest')

        # Apply mask to shared pattern
        perturbed = x_resized + self.delta.unsqueeze(0) * mask_up

        # Forward through backbone
        logits = self.backbone(perturbed)
        return logits