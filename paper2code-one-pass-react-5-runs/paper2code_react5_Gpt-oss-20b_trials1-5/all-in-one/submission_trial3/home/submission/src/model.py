"""
Simformer model: tokeniser → transformer → score prediction.
"""

import torch
import torch.nn as nn
from .tokenizer import Tokenizer
from .diffusion import TimeEmbedding
from .utils import random_attention_mask, identity_attention_mask, full_attention_mask


class Simformer(nn.Module):
    """
    Main Simformer network.
    """

    def __init__(
        self,
        param_dim: int,
        data_dim: int,
        embed_dim: int = 64,
        n_layers: int = 6,
        n_heads: int = 4,
        use_fourier: bool = False,
        fourier_dim: int = 32,
        attention_mask_type: str = "full",  # 'full', 'identity', 'random'
    ):
        super().__init__()
        self.tokenizer = Tokenizer(
            param_dim, data_dim, embed_dim, use_fourier, fourier_dim
        )
        self.time_emb = TimeEmbedding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 3,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.attention_mask_type = attention_mask_type

    def make_attention_mask(self, seq_len, device):
        if self.attention_mask_type == "full":
            return full_attention_mask(seq_len).to(device)
        if self.attention_mask_type == "identity":
            return identity_attention_mask(seq_len).to(device)
        if self.attention_mask_type == "random":
            return random_attention_mask(seq_len).to(device)
        raise ValueError(f"Unknown attention mask type {self.attention_mask_type}")

    def forward(self, theta, x, cond_mask, t):
        """
        Args:
            theta: [batch, param_dim]
            x:     [batch, data_dim]
            cond_mask: [batch, 2] binary mask (1 = conditioned)
            t:     [batch] diffusion time in [0,1]
        Returns:
            predicted_score: [batch, seq_len=2, embed_dim]
        """
        batch, _ = theta.shape
        seq_len = 2
        device = theta.device

        tokens = self.tokenizer(theta, x, cond_mask)  # [batch, 2, embed_dim]
        # Add time embedding
        time_emb = self.time_emb(t)  # [batch, embed_dim]
        tokens = tokens + time_emb.unsqueeze(1)

        # Transformer expects [seq_len, batch, embed_dim]
        tokens_t = tokens.permute(1, 0, 2)
        attn_mask = self.make_attention_mask(seq_len, device)
        # attn_mask: [seq_len, seq_len] bool – True where attention allowed
        # PyTorch expects mask where True disables attention
        attn_mask = ~attn_mask  # convert to disable mask
        output = self.transformer(tokens_t, src_key_padding_mask=None, mask=attn_mask)
        output = output.permute(1, 0, 2)  # back to [batch, seq_len, embed_dim]
        return output