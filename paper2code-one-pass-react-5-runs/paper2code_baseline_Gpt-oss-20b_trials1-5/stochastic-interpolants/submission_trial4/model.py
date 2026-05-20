import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sinusoidal_embedding

class SimpleUNet(nn.Module):
    """
    Very small UNet‑like architecture suitable for 28×28 MNIST.
    Input: image (1×28×28) + time embedding (channel‑wise).
    Output: velocity field (1×28×28).
    """
    def __init__(self, time_emb_dim=64, hidden_dim=64):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1 + hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x, t):
        """
        x: (B, 1, H, W)
        t: (B,) time scalar in [0,1]
        """
        B, C, H, W = x.shape
        time_emb = sinusoidal_embedding(t, dim=64)
        time_emb = self.time_emb(time_emb)  # (B, hidden_dim)
        time_emb = time_emb[:, :, None, None].expand(B, -1, H, W)

        # Concatenate time embedding to image
        inp = torch.cat([x, time_emb], dim=1)

        # Encoder
        e1 = self.enc1(inp)          # (B, hidden, H, W)
        e2 = self.enc2(e1)           # (B, hidden, H, W)

        # Bottleneck
        b = self.bottleneck(e2)

        # Decoder with skip connections
        d2 = torch.cat([b, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)  # (B, 1, H, W)
        return out