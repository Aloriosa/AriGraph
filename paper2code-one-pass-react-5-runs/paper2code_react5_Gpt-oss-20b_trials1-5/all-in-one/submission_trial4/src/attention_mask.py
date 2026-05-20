"""
Utility to create attention masks for the transformer.
Currently only provides the identity (fully connected) mask.
"""

import jax.numpy as jnp


def identity_mask(seq_len: int, batch_size: int) -> jnp.ndarray:
    """
    Returns a mask of shape (batch_size, seq_len, seq_len) with ones everywhere.
    """
    mask = jnp.ones((1, seq_len, seq_len))
    return jnp.broadcast_to(mask, (batch_size, seq_len, seq_len))