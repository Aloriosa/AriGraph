import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MaskGenerator(nn.Module):
    """
    Lightweight CNN that outputs a 3‑channel mask of the same spatial size
    as the resized input image. The architecture is intentionally small
    (1 conv -> 1 pool -> 2 conv -> 1 pool -> 1 conv).
    """
    def __init__(self, in_channels=3, out_channels=3, pool_times=2, latent_dim=64):
        super(MaskGenerator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, latent_dim, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(pool_times):
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(latent_dim, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        mask = self.net(x)
        mask = torch.sigmoid(mask)  # values in [0,1]
        return mask

class SMMReprogramming(nn.Module):
    """
    The core model that implements Sample‑Specific Multi‑Channel Masks (SMM).
    """
    def __init__(self, pretrained_backbone='resnet18', target_classes=10, patch_size=2):
        super(SMMReprogramming, self).__init__()
        self.backbone = getattr(models, pretrained_backbone)(pretrained=True)
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        # The backbone expects 224x224 input
        self.input_size = 224
        self.patch_size = patch_size

        # Trainable noise pattern shared across all samples
        self.delta = nn.Parameter(torch.zeros(3, self.input_size, self.input_size))
        # Mask generator
        self.mask_gen = MaskGenerator(in_channels=3, out_channels=3,
                                      pool_times=2, latent_dim=64)

        # Mapping from pretrained classes (0-999) to target classes
        self.mapping = None  # to be set externally

    def forward(self, x):
        """
        x: batch of images of shape (B,3,H,W) with H,W in {32,64,128}
        """
        device = x.device
        # Resize to 224x224
        resized = F.interpolate(x, size=(self.input_size, self.input_size),
                                mode='bilinear', align_corners=False)
        # Generate mask per sample
        mask = self.mask_gen(resized)  # shape (B,3,?,?)
        # If mask smaller due to pooling, upsample by nearest neighbor
        if mask.size(2) != self.input_size or mask.size(3) != self.input_size:
            mask = F.interpolate(mask, size=(self.input_size, self.input_size),
                                 mode='nearest')
        # Apply mask to noise pattern
        mod_noise = self.delta * mask
        # Reprogrammed input
        x_mod = resized + mod_noise
        # Forward through backbone
        logits = self.backbone(x_mod)
        # Apply output mapping
        if self.mapping is not None:
            logits = apply_mapping(logits, self.mapping)
        return logits

    def set_mapping(self, mapping):
        self.mapping = mapping