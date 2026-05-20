import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from utils import sigma_t, mu_t, target_score

class Tokenizer(nn.Module):
    """
    Tokenizer that turns each variable (parameter or data) into a token
    consisting of an identifier embedding, a value embedding and a
    condition‑state embedding.
    """
    def __init__(self,
                 num_vars: int,
                 embed_dim: int = 64,
                 ident_dim: int = 32,
                 cond_dim: int = 8):
        super().__init__()
        self.num_vars = num_vars
        self.ident_embed = nn.Embedding(num_vars, ident_dim)
        self.value_proj = nn.Linear(1, embed_dim)      # value is scalar
        self.cond_embed  = nn.Embedding(2, cond_dim)
        self.proj = nn.Linear(ident_dim + embed_dim + cond_dim, embed_dim)

    def forward(self, values: Tensor, cond: Tensor) -> Tensor:
        """
        values: (batch, seq_len, 1)  scalar values
        cond  : (batch, seq_len)    binary condition flag
        returns: (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = values.shape
        device = values.device
        ids = torch.arange(self.num_vars, device=device).unsqueeze(0).repeat(batch, 1)
        id_emb = self.ident_embed(ids)                     # (batch, seq_len, ident_dim)
        val_emb = self.value_proj(values)                  # (batch, seq_len, embed_dim)
        cond_emb = self.cond_embed(cond.long())            # (batch, seq_len, cond_dim)
        token = torch.cat([id_emb, val_emb, cond_emb], dim=-1)
        return self.proj(token)                            # (batch, seq_len, embed_dim)

class TransformerScoreModel(nn.Module):
    """
    Transformer that predicts the score for each token in the sequence.
    """
    def __init__(self,
                 seq_len: int,
                 embed_dim: int = 64,
                 num_layers: int = 6,
                 nhead: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=nhead,
                                                   dim_feedforward=embed_dim*4,
                                                   dropout=dropout,
                                                   activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_embed = nn.Linear(1, embed_dim)
        self.out_head   = nn.Linear(embed_dim, 1)          # score per token

    def forward(self, tokens: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        """
        tokens: (batch, seq_len, embed_dim)
        cond  : (batch, seq_len)  binary flag (unused in forward but kept for consistency)
        t     : (batch, 1)        continuous time [0,1]
        returns: (batch, seq_len)  score per token
        """
        b, seq_len, _ = tokens.size()
        t_emb = self.time_embed(t).unsqueeze(1).repeat(1, seq_len, 1)
        x = tokens + t_emb
        # Transformer expects (seq_len, batch, embed_dim)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return self.out_head(x).squeeze(-1)                # (batch, seq_len)

class Simformer(nn.Module):
    """
    Wrapper that contains the tokenizer and the score model.
    """
    def __init__(self,
                 num_vars: int,
                 seq_len: int,
                 embed_dim: int = 64,
                 num_layers: int = 6,
                 nhead: int = 4):
        super().__init__()
        self.tokenizer = Tokenizer(num_vars=num_vars, embed_dim=embed_dim)
        self.score_net = TransformerScoreModel(seq_len=seq_len,
                                               embed_dim=embed_dim,
                                               num_layers=num_layers,
                                               nhead=nhead)

    def forward(self, values: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        """
        values: (batch, seq_len, 1)
        cond  : (batch, seq_len)
        t     : (batch, 1)
        returns score: (batch, seq_len)
        """
        tokens = self.tokenizer(values, cond)
        return self.score_net(tokens, cond, t)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cpu'))