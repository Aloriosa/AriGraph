import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class Normalize(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.group_norm(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, with_adapter=False, adapter_dim=16, adapter_patch_size=4, adapter_num_heads=4, adapter_qkv_bias=True, adapter_drop=0.1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.with_adapter = with_adapter

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

        self.init_weights()
        if self.with_adapter:
            self.adapter = Adapter(in_channels=in_channels,
                                   out_channels=out_channels,
                                   temb_channels=temb_channels,
                                   embed_dim=adapter_dim,
                                   patch_size=adapter_patch_size,
                                   num_heads=adapter_num_heads,
                                   qkv_bias=adapter_qkv_bias,
                                   drop=adapter_drop)
        else:
            self.adapter = nn.Identity()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.0002)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        if self.with_adapter:
            adapter_out = self.adapter(h, temb)
            h = h + adapter_out

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, with_adapter=False, adapter_dim=16, adapter_patch_size=4, adapter_num_heads=4, adapter_qkv_bias=True, adapter_drop=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.with_adapter = with_adapter

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        if self.with_adapter:
            self.adapter = Adapter(in_channels=in_channels,
                                   out_channels=in_channels,
                                   embed_dim=adapter_dim,
                                   patch_size=adapter_patch_size,
                                   num_heads=adapter_num_heads,
                                   qkv_bias=adapter_qkv_bias,
                                   drop=adapter_drop)
        else:
            self.adapter = nn.Identity()

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        if self.with_adapter:
            adapter_out = self.adapter(h_, None)
            h_ = h_ + adapter_out

        return x + h_


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


class Model(nn.Module):
    def __init__(
            self,
            ch,
            in_channels,
            out_ch,
            ch_mult,
            num_res_blocks,
            attn_resolutions,
            dropout,
            resamp_with_conv,
            model_type,
            img_size,
            num_timesteps,
            with_adapter=False,
            adapter_dim=16,
            adapter_patch_size=4,
            adapter_num_heads=4,
            adapter_qkv_bias=True,
            adapter_drop=0.1
    ):
        super().__init__()

        if model_type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = img_size
        self.in_channels = in_channels
        self.with_adapter = with_adapter

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = img_size
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         with_adapter=self.with_adapter,
                                         adapter_dim=adapter_dim,
                                         adapter_patch_size=adapter_patch_size,
                                         adapter_num_heads=adapter_num_heads,
                                         adapter_qkv_bias=adapter_qkv_bias,
                                         adapter_drop=adapter_drop))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in,
                                          with_adapter=self.with_adapter,
                                          adapter_dim=adapter_dim,
                                          adapter_patch_size=adapter_patch_size,
                                          adapter_num_heads=adapter_num_heads,
                                          adapter_qkv_bias=adapter_qkv_bias,
                                          adapter_drop=adapter_drop))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       with_adapter=self.with_adapter,
                                       adapter_dim=adapter_dim,
                                       adapter_patch_size=adapter_patch_size,
                                       adapter_num_heads=adapter_num_heads,
                                       adapter_qkv_bias=adapter_qkv_bias,
                                       adapter_drop=adapter_drop)
        self.mid.attn_1 = AttnBlock(block_in,
                                    with_adapter=self.with_adapter,
                                    adapter_dim=adapter_dim,
                                    adapter_patch_size=adapter_patch_size,
                                    adapter_num_heads=adapter_num_heads,
                                    adapter_qkv_bias=adapter_qkv_bias,
                                    adapter_drop=adapter_drop)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       with_adapter=self.with_adapter,
                                       adapter_dim=adapter_dim,
                                       adapter_patch_size=adapter_patch_size,
                                       adapter_num_heads=adapter_num_heads,
                                       adapter_qkv_bias=adapter_qkv_bias,
                                       adapter_drop=adapter_drop)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         with_adapter=self.with_adapter,
                                         adapter_dim=adapter_dim,
                                         adapter_patch_size=adapter_patch_size,
                                         adapter_num_heads=adapter_num_heads,
                                         adapter_qkv_bias=adapter_qkv_bias,
                                         adapter_drop=adapter_drop))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in,
                                          with_adapter=self.with_adapter,
                                          adapter_dim=adapter_dim,
                                          adapter_patch_size=adapter_patch_size,
                                          adapter_num_heads=adapter_num_heads,
                                          adapter_qkv_bias=adapter_qkv_bias,
                                          adapter_drop=adapter_drop))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if self.with_adapter:
            self.end_adapter = Adapter(in_channels=block_in,
                                       out_channels=out_ch,
                                       temb_channels=None,
                                       embed_dim=adapter_dim,
                                       patch_size=adapter_patch_size,
                                       num_heads=adapter_num_heads,
                                       qkv_bias=adapter_qkv_bias,
                                       drop=adapter_drop)
        else:
            self.end_adapter = nn.Identity()

        if with_adapter:
            # freeze the base DDPM model when model with adapter
            for name, param in self.named_parameters():
                if "adapter" not in name.lower():
                    param.requires_grad = False

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.with_adapter:
            adapter_out = self.end_adapter(h, None)
        else:
            adapter_out = 0.0
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h + adapter_out