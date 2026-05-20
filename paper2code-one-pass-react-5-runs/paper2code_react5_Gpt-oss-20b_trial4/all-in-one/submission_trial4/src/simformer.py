"""
The Simformer model: a transformer‑based score estimator for the joint
distribution p(θ, x).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_dim: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Layer norm
        y = nn.LayerNorm()(x)
        # Self‑attention
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(y)
        # Residual
        x = x + y

        # Feed‑forward
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(self.hidden_dim)(y)
        # Residual
        return x + y


class Simformer(nn.Module):
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, token_embeds: jnp.ndarray,
                 t: jnp.ndarray,
                 deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            token_embeds: shape (B, seq_len, hidden_dim)
            t: shape (B,) diffusion time in [0,1]
        Returns:
            score: shape (B, seq_len, hidden_dim)
        """
        B, seq_len, _ = token_embeds.shape

        # Time embedding (sinusoidal)
        pe = self.sinusoidal_embedding(t, self.hidden_dim)
        pe = pe[:, None, :]  # (B, 1, hidden_dim)

        x = token_embeds + pe  # broadcast over seq_len

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=256,
                dropout_rate=self.dropout_rate,
            )(x, deterministic=deterministic)

        # Predict score for each token
        score = nn.Dense(self.hidden_dim, name="score_head")(x)
        return score

    @staticmethod
    def sinusoidal_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
        """
        Sinusoidal time embedding like in transformers.
        t: shape (B,) in [0, 1]
        Returns: shape (B, dim)
        """
        position = t[:, None]  # (B, 1)
        div_term = jnp.exp(jnp.arange(0, dim, 2) * -(jnp.log(10000.0) / dim))
        sin = jnp.sin(position * div_term)
        cos = jnp.cos(position * div_term)
        embed = jnp.concatenate([sin, cos], axis=1)
        if dim % 2 == 1:
            embed = jnp.pad(embed, ((0, 0), (0, 1)))
        return embed