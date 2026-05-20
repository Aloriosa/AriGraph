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
        self.down1 = self._block(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = self._block(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = self._block(base_channels * 4, base_channels * 8, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = self._block(base_channels * 8, base_channels * 8, time_emb_dim)
        
        # Upsample path
        self.up1 = self._block(base_channels * 16, base_channels * 4, time_emb_dim)
        self.up2 = self._block(base_channels * 8, base_channels * 2, time_emb_dim)
        self.up3 = self._block(base_channels * 4, base_channels, time_emb_dim)
        
        # Final convolution
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)
        
        # Downsample and upsample layers
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _block(self, in_channels, out_channels, time_emb_dim):
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
        
        # Initial conv
        x1 = self.init_conv(x)
        
        # Downsample path
        x2 = self.down1(x1)
        x2 = self.maxpool(x2)
        
        x3 = self.down2(x2)
        x3 = self.maxpool(x3)
        
        x4 = self.down3(x3)
        x4 = self.maxpool(x4)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Upsample path
        x = self.upsample(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up1(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        
        # Final conv
        x = self.final_conv(x)
        
        return x

class StochasticInterpolant(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=256):
        super(StochasticInterpolant, self).__init__()
        self.unet = UNet(in_channels, out_channels, base_channels, time_emb_dim)
        
    def forward(self, x_t, t):
        return self.unet(x_t, t)