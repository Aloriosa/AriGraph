# simformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Simformer(nn.Module):
    """
    Minimal Simformer implementation for the Gaussian‑linear toy task.
    Uses a vanilla Transformer encoder and a linear score head.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        Forward pass through the transformer.

        Args:
            tokens: (B, N, embed_dim)
            attn_mask: (N, N) with True at positions that should be masked

        Returns:
            scores: (B, N, embed_dim)
        """
        if attn_mask is not None:
            # Transformer expects mask of shape (B, N, N)
            attn_mask = attn_mask.unsqueeze(0).repeat(tokens.size(0), 1, 1)
        out = self.encoder(tokens, src_key_padding_mask=None, mask=attn_mask)
        return self.score_head(out)