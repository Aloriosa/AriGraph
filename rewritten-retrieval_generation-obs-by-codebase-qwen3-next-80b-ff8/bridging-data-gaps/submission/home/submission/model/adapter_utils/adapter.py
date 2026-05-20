import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 temb_channels=None,
                 embed_dim=16,
                 patch_size=4,
                 num_heads=4,
                 qkv_bias=True,
                 drop=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.temb_channels = temb_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_pooling = nn.AvgPool2d(patch_size, stride=patch_size)
        self.down_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)
        if in_channels % 2 == 0:
            self.norm1 = nn.GroupNorm(num_groups=int(in_channels / 2), num_channels=in_channels, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.BatchNorm2d(in_channels, eps=1e-6)
        self.up_conv = nn.Conv2d(embed_dim, out_channels or in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=int(embed_dim / 4), num_channels=embed_dim, eps=1e-6, affine=True)

        if temb_channels != None:
            self.temb_proj = nn.Linear(temb_channels, embed_dim)

        self.attention_block = AttentionBlock(dim=embed_dim,
                                              num_heads=num_heads,
                                              qkv_bias=qkv_bias,
                                              drop=drop,
                                              attn_drop=drop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.0002)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def nonlinearity(self, x):
        # swish
        return x * torch.sigmoid(x)

    def forward(self, x, temb=None):
        return self._forward(x, temb)

    def _forward(self, x, temb=None):
        B, C, H, W = x.shape
        ph = int(H / self.patch_size)
        pw = int(W / self.patch_size)

        assert H % self.patch_size == 0 and W % self.patch_size == 0, f"H:{H}, W:{W}, patch_size:{self.patch_size}"

        # Down sampling
        x = self.down_pooling(x)  # B C Ph Pw
        x = self.norm1(x)
        x = self.down_conv(x).flatten(2).transpose(1, 2)  # B Ph*Pw EC

        if temb != None:
            temb = self.temb_proj(self.nonlinearity(temb)).unsqueeze(dim=1)
            x = x + temb

        # Self-attention + MLP layers
        x = self.attention_block(x)

        # Up sampling
        x = x.reshape(B, int(ph * pw), self.embed_dim).transpose(1, 2)
        x = x.reshape(B, self.embed_dim, ph, pw)  # B EC Ph Pw
        x = torch.nn.functional.interpolate(x, scale_factor=self.patch_size, mode="nearest")  # B EC H W
        x = self.norm2(x)
        x = self.up_conv(x)  # B C H W
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x