"""
Tokenizer for the Simformer.
Each variable (θ or x) is represented by a token consisting of:
  * Value embedding (linear projection of the scalar)
  * Condition state embedding (one‑hot encoding of observed / unobserved)
  * Identifier embedding (shared for θ and x)
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


class Tokenizer(nn.Module):
    """
    Tokenizer that converts a batch of (θ, x) pairs and a condition mask
    into a token sequence ready for the transformer.
    """
    hidden_dim: int = 64
    condition_dim: int = 2  # 0 = unobserved, 1 = observed

    @nn.compact
    def __call__(self, theta: jnp.ndarray, x: jnp.ndarray,
                 mask: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            theta: shape (B, d_theta)
            x: shape (B, d_x)
            mask: shape (B, d_theta + d_x) where first d_theta columns
                  correspond to θ, last d_x to x. 1 indicates observed,
                  0 unobserved.
        Returns:
            token_embeds: shape (B, seq_len= d_theta + d_x, hidden_dim)
        """
        B = theta.shape[0]
        d_theta = theta.shape[1]
        d_x = x.shape[1]
        seq_len = d_theta + d_x

        # Value embeddings
        val_proj = nn.Dense(self.hidden_dim, use_bias=False, name="val_proj")
        theta_val = val_proj(theta)  # (B, d_theta, hidden_dim)
        x_val = val_proj(x)          # (B, d_x, hidden_dim)

        # Condition embeddings
        cond_proj = nn.Embed(self.condition_dim, self.hidden_dim,
                             name="cond_proj")
        mask_int = mask.astype(jnp.int32)
        theta_cond = cond_proj(mask_int[:, :d_theta])  # (B, d_theta, hidden_dim)
        x_cond = cond_proj(mask_int[:, d_theta:])     # (B, d_x, hidden_dim)

        # Identifier embeddings (shared: 0 for θ, 1 for x)
        id_proj = nn.Embed(2, self.hidden_dim,
                           name="id_proj")
        theta_id = id_proj(jnp.zeros((d_theta,), dtype=jnp.int32))  # (d_theta, hidden_dim)
        x_id = id_proj(jnp.ones((d_x,), dtype=jnp.int32))          # (d_x, hidden_dim)
        # Broadcast to batch
        theta_id = jnp.broadcast_to(theta_id, (B, d_theta, self.hidden_dim))
        x_id = jnp.broadcast_to(x_id, (B, d_x, self.hidden_dim))

        # Sum all embeddings
        theta_embed = theta_val + theta_cond + theta_id
        x_embed = x_val + x_cond + x_id

        # Concatenate along sequence dimension
        token_embeds = jnp.concatenate([theta_embed, x_embed], axis=1)
        return token_embeds