# tokenizer.py
import torch
import torch.nn as nn

class Tokenizer(nn.Module):
    """
    Very small tokenizer for the toy Gaussian‑linear task.
    Each variable (parameter θ or observation x) is represented by:
        - a learned identifier embedding
        - a linear projection of the raw value
        - a learned condition state embedding (0 = latent, 1 = conditioned)
    The resulting token embeddings are summed element‑wise.
    """
    def __init__(self, n_identifiers: int, embed_dim: int, value_dim: int = 32):
        super().__init__()
        self.id_emb = nn.Embedding(n_identifiers, embed_dim)
        self.value_proj = nn.Linear(value_dim, embed_dim)
        self.condition_emb = nn.Embedding(2, embed_dim)
        self.raw_value_proj = nn.Linear(1, value_dim)

    def forward(
        self,
        identifiers: torch.Tensor,  # (B, N) int64
        values: torch.Tensor,       # (B, N, 1) float
        cond_state: torch.Tensor,   # (B, N) int64 (0/1)
    ) -> torch.Tensor:
        """
        Build token embeddings.

        Args:
            identifiers: indices of variable types (θ or x)
            values: raw scalar values
            cond_state: 0 for latent, 1 for conditioned

        Returns:
            tokens: (B, N, embed_dim)
        """
        id_emb = self.id_emb(identifiers)          # (B,N,embed)
        raw = self.raw_value_proj(values)          # (B,N,embed)
        val_emb = self.value_proj(raw)             # (B,N,embed)
        cond_emb = self.condition_emb(cond_state)  # (B,N,embed)
        return id_emb + val_emb + cond_emb