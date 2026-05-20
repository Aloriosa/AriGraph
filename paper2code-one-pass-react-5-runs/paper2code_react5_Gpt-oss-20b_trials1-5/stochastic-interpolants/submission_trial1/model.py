import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------------------------------------------
# Simple sinusoidal time embedding
# ------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_freq: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq

    def forward(self, t: torch.Tensor):
        """
        t: (B,) in [0,1]
        Returns: (B, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_freq) * torch.arange(half, device=device) / half
        )
        args = t.unsqueeze(1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(t[:, :1])], dim=1)
        return emb

# ------------------------------------------------------------------
# Core velocity model (UNet‑like)
# ------------------------------------------------------------------
class SimpleUNet(nn.Module):
    """
    Very small UNet‑style network that takes the image, optional
    conditioning channels, and a time embedding.
    """
    def __init__(self, in_channels: int = 3, cond_channels: int = 0,
                 time_dim: int = 64, base_channels: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.cond_channels = cond_channels
        self.time_emb = TimeEmbedding(time_dim)

        # First conv receives image + condition + time embedding channels
        self.conv1 = nn.Conv2d(
            in_channels + cond_channels + time_dim,
            base_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None):
        """
        x: (B, C, H, W)
        t: (B,) in [0,1]
        cond: optional conditioning tensor
        """
        B, C, H, W = x.shape
        t_emb = self.time_emb(t).view(B, -1, 1, 1)

        if cond is not None:
            cond = cond.to(x.device)
            x_in = torch.cat([x, cond, t_emb], dim=1)
        else:
            x_in = torch.cat([x, t_emb], dim=1)

        h = F.relu(self.conv1(x_in))
        h = F.relu(self.conv2(h))
        out = self.conv3(h)
        return out