import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import get_time_embedding

# Simple 2‑D U‑Net block used in the paper
class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_cond):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=dim_out)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=dim_out)
        self.act2 = nn.SiLU()
        self.skip = nn.Identity() if dim_in == dim_out else nn.Conv2d(dim_in, dim_out, 1)
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(dim_cond, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        # Add time embedding
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h + self.skip(x)

class DownBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_cond, groups=8):
        super().__init__()
        self.res = ResBlock(dim_in, dim_out, dim_cond)
        self.down = nn.Conv2d(dim_out, dim_out, 3, stride=2, padding=1)

    def forward(self, x, time_emb):
        h = self.res(x, time_emb)
        return self.down(h), h

class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_cond, groups=8):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim_in, dim_out, 4, stride=2, padding=1)
        self.res = ResBlock(dim_out*2, dim_out, dim_cond)

    def forward(self, x, skip, time_emb):
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        return self.res(h, time_emb)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=256, dim_mults=(1,1,2,3,4),
                 resnet_groups=8, learned_sin=True, sin_dim=32,
                 attention_heads=4, attention_head_dim=64):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(1 if learned_sin else sin_dim, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch)
        )
        dims = [base_ch * m for m in dim_mults]
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        # Down
        for i in range(len(dims)-1):
            self.down_blocks.append(DownBlock(dims[i], dims[i+1], base_ch))
        # Bottleneck
        self.bottleneck = ResBlock(dims[-1], dims[-1], base_ch)
        # Up
        for i in range(len(dims)-1, 0, -1):
            self.up_blocks.append(UpBlock(dims[i], dims[i-1], base_ch))
        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        """
        x: [B, C, H, W]
        t: [B]  (time scalar in [0,1])
        """
        # Time embedding
        if self.time_mlp is None:
            time_emb = torch.zeros(x.shape[0], self.time_mlp[0].out_features,
                                   device=x.device)
        else:
            time_emb = get_time_embedding(t, dim=self.time_mlp[0].in_features)
        h = self.in_conv(x)
        skips = []

        # Down
        for block in self.down_blocks:
            h, skip = block(h, time_emb)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck(h, time_emb)

        # Up
        for block in self.up_blocks:
            skip = skips.pop()
            h = block(h, skip, time_emb)

        return self.out_conv(h)

# Helper to instantiate the model
def create_velocity_model(config):
    return UNet(
        in_ch=3,
        out_ch=3,
        base_ch=config['unet_channels'],
        dim_mults=config['unet_dim_mults'],
        resnet_groups=config['unet_resnet_groups'],
        learned_sin=config['unet_learned_sin'],
        sin_dim=config['unet_sin_dim'],
        attention_heads=config['unet_attention_heads'],
        attention_head_dim=config['unet_attention_head_dim']
    )