import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """A simple conv → BN → ReLU block."""
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ks,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UNetSmall(nn.Module):
    """
    Very small U‑Net used for velocity prediction.
    Input channels: 3 (image) + 1 (mask) + 1 (time) = 5
    Output channels: 3 (velocity)
    """
    def __init__(self, in_channels=5, base_ch=32, num_levels=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool2d(2)

        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # BxC x32x32
        e2 = self.enc2(self.pool(e1))  # Bx2C x16x16
        e3 = self.enc3(self.pool(e2))  # Bx4C x8x8

        # Bottleneck skip
        d3 = self.dec3(torch.cat([e3, F.interpolate(e2, size=e3.shape[-2:], mode='nearest')], dim=1))
        d2 = self.dec2(torch.cat([d3, F.interpolate(e1, size=d3.shape[-2:], mode='nearest')], dim=1))
        d1 = self.dec1(d2)

        return self.out_conv(d1)  # Bx3 x32x32