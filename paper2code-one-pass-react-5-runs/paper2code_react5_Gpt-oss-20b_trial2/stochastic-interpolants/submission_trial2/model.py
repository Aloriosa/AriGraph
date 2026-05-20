import torch
import torch.nn as nn
import math


# ----------------------------------------------------------------------
# Time embedding (sinusoidal)
# ----------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding used in diffusion/ODE models.
    Input: t in [0,1] of shape (B,)
    Output: embedding of shape (B, dim)
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) float tensor in [0,1]
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )
        t = t[:, None] * emb_scale[None, :]          # (B, half_dim)
        emb = torch.cat([torch.sin(t), torch.cos(t)], dim=1)  # (B, dim)
        return emb


# ----------------------------------------------------------------------
# Convolutional building blocks
# ----------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ----------------------------------------------------------------------
# UNet with time embedding and optional conditioning
# ----------------------------------------------------------------------
class UNet(nn.Module):
    """
    A lightweight 3‑level UNet that accepts an optional conditioning
    tensor.  The conditioning channels are concatenated to the input
    feature map at every step.
    """
    def __init__(
        self,
        base_channels: int = 3,
        cond_channels: int = 0,
        out_channels: int = 3,
        time_emb_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        in_channels = base_channels + cond_channels
        self.input_conv = ConvBlock(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  – input image (base + conditioning)
        t: (B,) float in [0,1]
        """
        B, C, H, W = x.shape
        t_emb = self.time_embed(t)          # (B, time_emb_dim*4)
        t_emb = t_emb[:, :, None, None]     # broadcast to (B,*,H,W)
        h = self.input_conv(x + t_emb)
        d1 = self.down1(h)
        d2 = self.down2(d1)
        b = self.bottleneck(d2)
        u1 = self.up1(b, d2)
        u2 = self.up2(u1, d1)
        u3 = self.up3(u2, h)
        out = self.out_conv(u3)
        return out