"""Simple Transformer‑style MLP score model for diffusion."""

import haiku as hk
import jax.numpy as jnp
from typing import Tuple

class Tokenizer(hk.Module):
    """
    Convert each variable (parameter or data) into a token.
    Token = [identity (one‑hot), value, condition flag].
    """
    def __init__(self, n_vars: int, embedding_dim: int = 32, name=None):
        super().__init__(name=name)
        self.n_vars = n_vars
        self.embedding_dim = embedding_dim

    def __call__(self, values: jnp.ndarray, cond_mask: jnp.ndarray) -> jnp.ndarray:
        """
        values: shape (batch, n_vars)
        cond_mask: shape (batch, n_vars) – 1 if observed, 0 if latent
        """
        batch_size = values.shape[0]
        # Identity embedding
        identity = jnp.eye(self.n_vars, dtype=jnp.float32)  # (n_vars, n_vars)
        identity = jnp.tile(identity[None, :, :], (batch_size, 1, 1))  # (batch, n_vars, n_vars)
        # Condition flag embedding
        cond_emb = hk.Embed(2, self.embedding_dim)(cond_mask.astype(jnp.int32))  # (batch, n_vars, d)
        # Value projection
        val_proj = hk.Linear(self.embedding_dim, name="val_proj")

        values_proj = val_proj(values[..., None])  # (batch, n_vars, d)

        tokens = jnp.concatenate([identity, values_proj, cond_emb], axis=-1)  # (batch, n_vars, d+ n_vars + d)
        return tokens


class SimpleTransformer(hk.Module):
    """
    Very small Transformer encoder that operates on the token sequence.
    """
    def __init__(self,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 hidden_dim: int = 128,
                 name=None):
        super().__init__(name=name)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """
        tokens: (batch, seq_len, token_dim)
        returns: (batch, seq_len, token_dim)
        """
        seq_len = tokens.shape[1]
        h = tokens
        for i in range(self.n_layers):
            # Multi‑head self‑attention
            attn = hk.MultiHeadAttention(
                num_heads=self.n_heads,
                key_size=self.hidden_dim // self.n_heads,
                model_size=self.hidden_dim,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                name=f"mh_attn_{i}"
            )
            attn_out = attn(h, h, h)
            h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h + attn_out)

            # Feed‑forward
            ff = hk.Sequential([
                hk.Linear(4 * self.hidden_dim, name=f"ff1_{i}"),
                jax.nn.relu,
                hk.Linear(self.hidden_dim, name=f"ff2_{i}")
            ])
            ff_out = ff(h)
            h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h + ff_out)

        return h


class ScoreModel(hk.Module):
    """
    Main score model: Tokenizer → Transformer → linear head → score per variable.
    """
    def __init__(self,
                 n_vars: int,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 hidden_dim: int = 128,
                 name=None):
        super().__init__(name=name)
        self.n_vars = n_vars
        self.tokenizer = Tokenizer(n_vars=n_vars, embedding_dim=hidden_dim//2, name="tokenizer")
        self.transformer = SimpleTransformer(n_layers=n_layers,
                                            n_heads=n_heads,
                                            hidden_dim=hidden_dim,
                                            name="transformer")
        self.head = hk.Linear(self.n_vars, name="score_head")  # score per variable

    def __call__(self,
                 values: jnp.ndarray,
                 cond_mask: jnp.ndarray) -> jnp.ndarray:
        """
        values: (batch, n_vars)
        cond_mask: (batch, n_vars) – 1 if observed, 0 if latent
        returns: (batch, n_vars) – predicted score for each variable
        """
        tokens = self.tokenizer(values, cond_mask)  # (batch, n_vars, token_dim)
        h = self.transformer(tokens)                # (batch, n_vars, hidden_dim)
        scores = self.head(h)                       # (batch, n_vars, n_vars)
        # We only need the diagonal: score for each variable
        scores = jnp.diagonal(scores, axis1=1, axis2=2)  # (batch, n_vars)
        return scores