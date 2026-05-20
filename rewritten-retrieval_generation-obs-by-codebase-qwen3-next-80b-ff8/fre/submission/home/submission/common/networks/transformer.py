import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence

class MlpBlock(nn.Module):
    mlp_dim: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        x = nn.Dense(self.mlp_dim, dtype=self.dtype)(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(inputs.shape[-1], dtype=self.dtype)(x)
        return x

class Encoder1DBlock(nn.Module):
    mlp_dim: int
    num_heads: int
    causal: bool
    dropout_rate: float
    attention_dropout_rate: float
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, deterministic, train=True):
        if self.causal:
            causal_mask = nn.make_causal_mask(jnp.ones((inputs.shape[0], inputs.shape[1]), dtype="bool"), dtype="bool")
        else:
            causal_mask = None

        # Attention block.
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            decode=False,
            num_heads=self.num_heads)(x, x, causal_mask)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(y, deterministic=deterministic)

        return x + y

class Transformer(nn.Module):
    num_layers: int
    emb_dim: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float
    attention_dropout_rate: float
    causal: bool = True

    @nn.compact
    def __call__(self, x, *, train):
        assert x.ndim == 3  # (batch, len, emb)
        assert x.shape[-1] == self.emb_dim

        # Input Encoder. Each layer processes x, but the shape of x does not change.
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                    mlp_dim=self.mlp_dim,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    name=f'encoderblock_{lyr}',
                    causal=self.causal,
                    num_heads=self.num_heads)(
                            x, deterministic=not train, train=train)
        encoded = nn.LayerNorm(name='encoder_norm')(x)

        return encoded