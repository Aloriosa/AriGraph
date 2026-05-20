import torch
import torch.nn as nn
from src.arch.attention_mask_builder import build_attention_mask
from src.data.tokenizer import Tokenizer

class TransformerScoreModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super(TransformerScoreModel, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, d_model))  # Max sequence length 512
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Linear(d_model, 1)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # tokens: (batch_size, seq_len)
        # mask: (batch_size, seq_len)
        # t: (batch_size,)
        
        batch_size, seq_len = tokens.shape
        
        # Embed tokens
        x = self.embedding(tokens)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]  # Broadcast positional encoding
        
        # Embed time step
        t_emb = self.time_embedding(t.unsqueeze(1))  # (batch_size, d_model)
        t_emb = t_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Add time embedding to first position (as in DiT)
        x[:, 0, :] = x[:, 0, :] + t_emb.squeeze(1)
        
        # Build attention mask using the provided mask
        # Note: The attention_mask_builder expects sim_metadata, but we only have mask
        # We simulate sim_metadata with the provided mask
        sim_metadata = {
            'attention_mask': mask,
            'seq_len': seq_len,
            'batch_size': batch_size
        }
        attention_mask = build_attention_mask(sim_metadata)  # (batch_size, seq_len, seq_len)
        
        # Pass through transformer
        x = self.transformer_encoder(x, src_key_padding_mask=~mask)  # mask is boolean: True for valid tokens
        
        # Compute per-token scores
        scores = self.score_head(x).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask to ignore padding tokens
        scores = scores.masked_fill(~mask, 0.0)
        
        return scores