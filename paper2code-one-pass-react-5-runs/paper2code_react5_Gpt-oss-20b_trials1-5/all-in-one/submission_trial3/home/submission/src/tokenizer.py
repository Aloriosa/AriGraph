"""
Tokenizer for Simformer.

Each variable (parameter or data) is represented as a token:
    token = id_embedding + value_embedding + cond_embedding
The value embedding can be a linear projection of the raw value(s)
or a Fourier embedding for function‑valued parameters.
"""

import torch
import torch.nn as nn
import math
from .utils import fourier_features


class Tokenizer(nn.Module):
    def __init__(
        self,
        param_dim: int,
        data_dim: int,
        embed_dim: int = 64,
        use_fourier: bool = False,
        fourier_dim: int = 32,
    ):
        """
        Args:
            param_dim: number of parameters (scalar or vector)
            data_dim: number of data points (scalar or vector)
            embed_dim: dimension of token embeddings
            use_fourier: if True, use random Fourier features for value embedding
            fourier_dim: dimensionality of Fourier features
        """
        super().__init__()
        self.param_dim = param_dim
        self.data_dim = data_dim
        self.embed_dim = embed_dim
        self.use_fourier = use_fourier
        self.fourier_dim = fourier_dim

        # Identifier embeddings: one per variable (param, data)
        self.id_emb = nn.Parameter(
            torch.randn(2, embed_dim) * 0.02
        )  # [2, embed_dim] – [param, data]

        # Conditional state embeddings: [latent=0, conditioned=1]
        self.cond_emb = nn.Parameter(
            torch.randn(2, embed_dim) * 0.02
        )  # [2, embed_dim]

        # Value embeddings
        if use_fourier:
            # For Fourier, we store omega and b as buffers
            self.register_buffer(
                "fourier_omega",
                torch.randn(fourier_dim, dtype=torch.float32),
            )
            self.register_buffer(
                "fourier_b",
                torch.rand(fourier_dim, dtype=torch.float32) * 2 * math.pi,
            )
            self.value_proj = nn.Linear(fourier_dim, embed_dim, bias=False)
        else:
            self.value_proj = nn.Linear(
                max(param_dim, data_dim), embed_dim, bias=False
            )

    def forward(self, theta, x, cond_mask):
        """
        Args:
            theta: [batch, param_dim]  (params)
            x:     [batch, data_dim]   (data)
            cond_mask: [batch, 2]  binary mask where 1 = conditioned (observed)
                       first entry for params, second for data
        Returns:
            tokens: [batch, seq_len=2, embed_dim]
        """
        batch = theta.shape[0]
        # Value embeddings
        if self.use_fourier:
            # For each variable, we flatten the vector and apply Fourier
            theta_feat = fourier_features(theta, self.fourier_dim)
            x_feat = fourier_features(x, self.fourier_dim)
            theta_val = self.value_proj(theta_feat)
            x_val = self.value_proj(x_feat)
        else:
            theta_val = self.value_proj(theta)
            x_val = self.value_proj(x)

        # Identifier embeddings
        id_emb = self.id_emb.unsqueeze(0).expand(batch, -1, -1)  # [batch, 2, embed_dim]

        # Conditional embeddings
        cond_emb = self.cond_emb[cond_mask.long()].unsqueeze(1)  # [batch, 1, 2, embed_dim] -> [batch,2,embed_dim]
        cond_emb = cond_emb.squeeze(1)

        # Combine
        tokens = id_emb + torch.stack([theta_val, x_val], dim=1) + cond_emb
        return tokens