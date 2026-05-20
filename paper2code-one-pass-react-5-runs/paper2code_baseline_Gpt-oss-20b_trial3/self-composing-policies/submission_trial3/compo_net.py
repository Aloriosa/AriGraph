"""
CompoNet – Self‑composing policy network.

The network grows one module per task.  Each module is a
Self‑Composing Policy Module (see Fig. 2 in the paper) that
consists of:

  * Output attention head
  * Input attention head
  * Internal policy (MLP)

All modules share the same hidden dimension `hidden_dim` and
action dimension `action_dim`.  Previous modules are frozen
after a task is finished.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


class SelfComposeModule(nn.Module):
    """
    Self‑Composing Policy Module.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of hidden state (d_model in the paper).
    action_dim : int
        Number of actions (|A|).
    """
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # ---- Output attention head ----
        self.out_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_k = nn.Linear(action_dim, hidden_dim, bias=False)
        # value matrix is the raw policy outputs from previous modules
        # no learnable weights

        # ---- Input attention head ----
        self.in_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.in_k = nn.Linear(action_dim+action_dim, hidden_dim, bias=False)
        self.in_v = nn.Linear(action_dim+action_dim, hidden_dim, bias=False)

        # ---- Internal policy (MLP) ----
        self.internal = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, h_s: torch.Tensor,
                prev_policies: torch.Tensor):
        """
        Forward pass for one module.

        Parameters
        ----------
        h_s : torch.Tensor of shape (batch, hidden_dim)
            Encoded state vector from the encoder.
        prev_policies : torch.Tensor of shape
            (batch, n_prev, action_dim)
            The output logits of all previous modules.

        Returns
        -------
        logits : torch.Tensor of shape (batch, action_dim)
            The policy logits for the current module.
        """
        batch = h_s.size(0)
        n_prev = prev_policies.size(1)

        # --- Output attention head ---
        q_out = self.out_q(h_s)                     # (B, H)
        k_out = self.out_k(prev_policies.reshape(batch*n_prev, -1))
        k_out = k_out.reshape(batch, n_prev, self.hidden_dim)
        # value matrix is prev_policies itself
        v_out = prev_policies                          # (B, N, A)

        # attention weights
        att_out = F.softmax(
            torch.matmul(q_out.unsqueeze(1), k_out.transpose(-1, -2))
            / math.sqrt(self.hidden_dim), dim=-1)      # (B,1,N)
        # weighted sum over previous policies
        v_out = torch.sum(att_out * v_out, dim=1)      # (B, A)

        # --- Input attention head ---
        # concatenate output attention result and previous policies
        inp = torch.cat([v_out.unsqueeze(1), prev_policies], dim=1)
        # inp shape (B, N+1, A)
        q_in = self.in_q(h_s)                         # (B, H)
        k_in = self.in_k(inp.reshape(batch*(n_prev+1), -1))
        k_in = k_in.reshape(batch, n_prev+1, self.hidden_dim)
        v_in = self.in_v(inp.reshape(batch*(n_prev+1), -1))
        v_in = v_in.reshape(batch, n_prev+1, self.hidden_dim)

        att_in = F.softmax(
            torch.matmul(q_in.unsqueeze(1), k_in.transpose(-1, -2))
            / math.sqrt(self.hidden_dim), dim=-1)      # (B,1,N+1)
        v_in = torch.sum(att_in * v_in, dim=1)          # (B, H)

        # --- Internal policy ---
        inp_policy = torch.cat([v_out, v_in], dim=-1)   # (B, A+H)
        logits = self.internal(inp_policy)             # (B, A)

        # Final output: add v_out (from output head) to logits
        logits = logits + v_out
        return logits


class CompoNet(nn.Module):
    """
    The full CompoNet actor.

    It keeps a list of modules, one per task.  When a new task arrives
    a new module is appended and all previously frozen modules are
    frozen (requires_grad=False).  The encoder is a simple MLP for
    continuous tasks or a small CNN for Atari games – the same encoder
    is used for all modules.
    """
    def __init__(self, input_dim: int, action_dim: int,
                 hidden_dim: int, encoder_type: str = "mlp"):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.modules_list = nn.ModuleList()

        # Encoder – a tiny MLP or CNN
        if encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        else:  # cnn
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),  # (3,210,160)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, hidden_dim),
                nn.ReLU()
            )

    def add_module(self):
        """Create a new self‑composing module and freeze all old ones."""
        if len(self.modules_list) > 0:
            for m in self.modules_list:
                for p in m.parameters():
                    p.requires_grad = False
        new_mod = SelfComposeModule(self.hidden_dim, self.action_dim)
        self.modules_list.append(new_mod)

    def forward(self, x):
        """
        Forward pass for the whole CompoNet actor.

        Parameters
        ----------
        x : torch.Tensor
            Raw observation.  For Atari it is (B,3,210,160).
            For Meta‑World it is (B,39).

        Returns
        -------
        logits : torch.Tensor of shape (B, action_dim)
            Final policy logits for the current task.
        """
        h_s = self.encoder(x)          # (B, H)
        # Gather outputs of all previous modules
        prev_logit_tensors = []
        for mod in self.modules_list:
            logits_prev = mod(h_s, prev_logit_tensors[-1:] if prev_logit_tensors else torch.empty(0))
            prev_logit_tensors.append(logits_prev)
        if not prev_logit_tensors:
            # no previous module – just use a random linear layer
            dummy = nn.Linear(self.hidden_dim, self.action_dim).to(x.device)
            return dummy(h_s)
        prev_stack = torch.stack(prev_logit_tensors, dim=1)   # (B, N, A)
        # Current module (the last in the list)
        cur_mod = self.modules_list[-1]
        logits = cur_mod(h_s, prev_stack)
        return logits