import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256):
        super(UNet, self).__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsample path
        self.down1 = self._block(base_channels, base_channels * 2)
        self.down2 = self._block(base_channels * 2, base_channels * 4)
        self.down3 = self._block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self._block(base_channels * 8, base_channels * 16)
        
        # Upsample path
        self.up1 = self._block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = self._block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = self._block(base_channels * 4 + base_channels * 2, base_channels * 2)
        
        # Final convolution
        self.final_conv = nn.Conv2d(base_channels * 2, out_channels, 1)
        
        # Attention mechanisms
        self.attention1 = SelfAttention(base_channels * 4)
        self.attention2 = SelfAttention(base_channels * 8)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsample
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Upsample with attention
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.attention1(x)
        x = self.up2(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.attention2(x)
        x = self.up3(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = x.view(batch_size, channels, h * w).permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1).view(batch_size, channels, h, w)