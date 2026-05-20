import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfComposingModule(nn.Module):
    """
    A single self‑composing policy module.
    Implements:
      * Output attention head
      * Input attention head
      * Internal MLP policy
    """

    def __init__(self, state_dim: int, action_dim: int, d_model: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Output attention head
        self.q_out = nn.Linear(state_dim, d_model)
        self.k_out = nn.Linear(action_dim, d_model)

        # Input attention head
        self.q_in = nn.Linear(state_dim, d_model)
        # Key size: concatenated v_out + all prev outputs
        self.k_in = nn.Linear(action_dim + action_dim, d_model)
        self.v_in = nn.Linear(action_dim + action_dim, d_model)

        # Internal policy (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, state: torch.Tensor, prev_outputs: torch.Tensor):
        """
        Parameters
        ----------
        state : Tensor of shape (batch, state_dim)
        prev_outputs : Tensor of shape (batch, n_prev, action_dim)

        Returns
        -------
        action_tensor : Tensor of shape (batch, action_dim)
        """
        batch = state.shape[0]
        n_prev = prev_outputs.shape[1]

        # ----- Output attention head -----
        q_out = self.q_out(state)  # (B, d_model)
        if n_prev > 0:
            K_out = self.k_out(prev_outputs)  # (B, n_prev, d_model)
            V_out = prev_outputs  # (B, n_prev, action_dim)

            # Scaled dot‑product attention
            attn_logits = torch.bmm(q_out.unsqueeze(1), K_out.transpose(1, 2)) / math.sqrt(
                self.d_model
            )  # (B, 1, n_prev)
            attn = F.softmax(attn_logits, dim=-1)  # (B, 1, n_prev)
            v_out = torch.bmm(attn, V_out).squeeze(1)  # (B, action_dim)
        else:
            v_out = torch.zeros(batch, self.action_dim, device=state.device)

        # ----- Input attention head -----
        # Concatenate v_out (B,1,act) with prev_outputs (B,n_prev,act)
        if n_prev > 0:
            P = torch.cat([v_out.unsqueeze(1), prev_outputs], dim=1)  # (B, n_prev+1, act)
        else:
            P = v_out.unsqueeze(1)  # (B,1,act)

        q_in = self.q_in(state)  # (B, d_model)
        K_in = self.k_in(P)  # (B, n_prev+1, d_model)
        V_in = self.v_in(P)  # (B, n_prev+1, d_model)

        attn_in_logits = torch.bmm(q_in.unsqueeze(1), K_in.transpose(1, 2)) / math.sqrt(
            self.d_model
        )  # (B,1,n_prev+1)
        attn_in = F.softmax(attn_in_logits, dim=-1)  # (B,1,n_prev+1)
        inp = torch.bmm(attn_in, V_in).squeeze(1)  # (B, d_model)

        # ----- Internal policy -----
        mlp_out = self.mlp(torch.cat([state, inp], dim=-1))  # (B, action_dim)

        # ----- Final output -----
        return v_out + mlp_out  # (B, action_dim)