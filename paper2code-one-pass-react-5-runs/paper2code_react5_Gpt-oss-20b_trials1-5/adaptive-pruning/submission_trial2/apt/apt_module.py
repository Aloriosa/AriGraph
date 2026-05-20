"""
Custom modules implementing LoRA and pruning support.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LoRALinear(nn.Module):
    """
    A linear layer with a low‑rank LoRA adapter.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 4,
                 scaling: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = scaling
        # Base weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # LoRA weights
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)
        lora = self.scaling * F.linear(x, F.linear(x, self.A.t()).t() @ self.B.t())
        return base + lora

    def expand_rank(self, new_r: int):
        """
        Increase the LoRA rank by concatenating new parameters.
        """
        if new_r <= self.r:
            return
        # New A and B
        new_A = nn.Parameter(torch.randn(new_r - self.r, self.in_features) * 0.01)
        new_B = nn.Parameter(torch.randn(self.out_features, new_r - self.r) * 0.01)
        self.A = nn.Parameter(torch.cat([self.A, new_A], dim=0))
        self.B = nn.Parameter(torch.cat([self.B, new_B], dim=1))
        self.r = new_r

class PrunableSelfAttention(nn.Module):
    """
    Wraps DistilBertSelfAttention to support head pruning.
    """
    def __init__(self, attn_module: nn.Module):
        super().__init__()
        # Copy all sub‑modules
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        self.q_proj = attn_module.q_proj
        self.k_proj = attn_module.k_proj
        self.v_proj = attn_module.v_proj
        self.out_proj = attn_module.out_proj
        self.dropout = attn_module.dropout
        # Head mask (1 = kept, 0 = pruned)
        self.head_mask = nn.Parameter(torch.ones(self.num_heads, dtype=torch.bool),
                                     requires_grad=False)

    def forward(self, hidden_states, attention_mask=None):
        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        batch_size, seq_len, _ = hidden_states.size()

        # Reshape for multi‑head
        def reshape(x):
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        # Apply head mask
        mask = self.head_mask.view(1, 1, -1, 1)  # broadcast
        q = q * mask
        k = k * mask
        v = v * mask

        # Scaled dot‑product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq, heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, -1)

        attn_output = self.out_proj(attn_output)
        return attn_output