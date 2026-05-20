import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim_out)
        if dim_in != dim_out:
            self.res_conv = nn.Conv2d(dim_in, dim_out, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if self.res_conv is not None:
            x = self.res_conv(x)
        return F.relu(h + x)

class UNetVelocity(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=64):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(time_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.enc1 = ResidualBlock(in_channels+1, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels*2)
        self.enc3 = ResidualBlock(base_channels*2, base_channels*4)
        self.dec3 = ResidualBlock(base_channels*4, base_channels*2)
        self.dec2 = ResidualBlock(base_channels*2, base_channels)
        self.dec1 = ResidualBlock(base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, 3, 1)

    def forward(self, x, t, mask=None):
        """
        x: (B, C, H, W) interpolated sample
        t: (B,) scalar in [0,1]
        mask: optional (B, 1, H, W) mask used for conditioning
        """
        # Time embedding
        t_emb = self.time_emb(t[:, None])  # B×128
        t_emb = t_emb[:, :, None, None]
        if mask is not None:
            inp = torch.cat([x, mask], dim=1)
        else:
            inp = x
        h1 = self.enc1(inp)
        h2 = self.enc2(F.avg_pool2d(h1, 2))
        h3 = self.enc3(F.avg_pool2d(h2, 2))
        # Upsample
        d3 = self.dec3(F.interpolate(h3, scale_factor=2))
        d2 = self.dec2(F.interpolate(d3 + h2, scale_factor=2))
        d1 = self.dec1(F.interpolate(d2 + h1, scale_factor=2))
        return self.out(d1)  # B×3×H×W