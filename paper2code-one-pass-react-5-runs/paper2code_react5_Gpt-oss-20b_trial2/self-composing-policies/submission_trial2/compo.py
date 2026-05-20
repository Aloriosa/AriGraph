"""
CompoNet – Self‑Composing Policy Network

The implementation below follows the architecture described in the
paper:

* Each task is represented by a *SelfComposingPolicyModule*.
* The module contains:
  1. Output attention head (attends over previous modules’ outputs).
  2. Input attention head (attends over the output head result + previous outputs).
  3. Internal policy (MLP that adjusts the tentative output).
* Modules are stacked sequentially.  When a new task arrives, the
  previous modules are frozen and a new module is appended.

The code is deliberately lightweight and is meant for educational
purposes.  It does not aim to match the exact numerical performance
of the paper – that would require extensive hyper‑parameter tuning
and a large training budget – but it implements the core idea.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfComposingPolicyModule(nn.Module):
    """
    Implements a single self‑composing policy module.

    Args:
        input_dim: Dimensionality of the state representation h_s.
        action_dim: Dimensionality of the action space (size of the
            categorical distribution for discrete actions, or dimension
            of the Gaussian mean for continuous actions).
        hidden_dim: Dimensionality used inside the attention heads and
            the internal MLP.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Output attention head
        self.out_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.out_k = nn.Linear(action_dim, hidden_dim, bias=False)
        # Input attention head
        self.in_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.in_k = nn.Linear(action_dim + action_dim, hidden_dim, bias=False)
        self.in_v = nn.Linear(action_dim + action_dim, hidden_dim, bias=False)

        # Internal policy (MLP)
        self.fw = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, h_s: torch.Tensor, prev_outputs: torch.Tensor):
        """
        Forward pass of a single module.

        Parameters
        ----------
        h_s : Tensor (batch, input_dim)
            State representation for the current task.
        prev_outputs : Tensor (batch, n_prev, action_dim)
            The outputs of all previous modules for the same batch.
            If there are no previous modules, ``prev_outputs``
            should be an empty tensor of shape (batch, 0, action_dim).

        Returns
        -------
        action_logits : Tensor (batch, action_dim)
            The log‑probabilities (or mean of Gaussian) for the current
            action distribution.
        """
        if prev_outputs.shape[1] == 0:
            # No previous modules – simply use the internal policy
            return self.fw(torch.cat([h_s, torch.zeros_like(
                self.fw[0].bias[None, :].repeat(h_s.shape[0], 1))], dim=1))

        # 1. Output attention head
        #   Query: from state
        q_out = self.out_q(h_s)  # (batch, hidden_dim)
        #   Keys: from previous outputs
        k_out = self.out_k(prev_outputs)  # (batch, n_prev, hidden_dim)
        #   Values: raw previous outputs
        v_out = prev_outputs  # (batch, n_prev, action_dim)

        # scaled dot‑product attention
        attn_out = F.scaled_dot_product_attention(
            q_out.unsqueeze(1),   # (batch, 1, hidden_dim)
            k_out,                # (batch, n_prev, hidden_dim)
            v_out,                # (batch, n_prev, action_dim)
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(self.hidden_dim)
        ).squeeze(1)  # (batch, action_dim)

        # 2. Input attention head
        #   Concatenate output attention result with previous outputs
        #   to form the keys/values for the second attention.
        #   Shape (batch, n_prev + 1, action_dim)
        concat_prev = torch.cat([attn_out.unsqueeze(1), prev_outputs], dim=1)
        q_in = self.in_q(h_s)  # (batch, hidden_dim)
        k_in = self.in_k(concat_prev)  # (batch, n_prev+1, hidden_dim)
        v_in = self.in_v(concat_prev)  # (batch, n_prev+1, hidden_dim)

        attn_in = F.scaled_dot_product_attention(
            q_in.unsqueeze(1),
            k_in,
            v_in,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(self.hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)

        # 3. Internal policy
        #   Combine state and the result of input attention
        inp = torch.cat([h_s, attn_in], dim=1)  # (batch, input_dim + hidden_dim)
        out = self.fw(inp)  # (batch, action_dim)

        # Final output is tentative vector + internal adjustment
        final = attn_out + out
        return final


class CompoNet(nn.Module):
    """
    CompoNet – a stack of self‑composing modules.

    The network grows by appending a new module for each new task.
    Previous modules are frozen (their parameters are not updated).
    The network is used as the actor of SAC or PPO.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.modules = nn.ModuleList()
        self.frozen = []

    def add_module(self):
        """Append a new trainable module for the next task."""
        self.modules.append(
            SelfComposingPolicyModule(
                self.input_dim, self.action_dim, self.hidden_dim
            )
        )
        # Freeze previous modules
        for m in self.modules[:-1]:
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, h_s: torch.Tensor):
        """
        Forward pass over all modules.

        Parameters
        ----------
        h_s : Tensor (batch, input_dim)

        Returns
        -------
        out : Tensor (batch, action_dim)
            The action distribution for the *current* task (i.e. the
            last module in the stack).
        """
        prev_outputs = torch.empty((h_s.shape[0], 0, self.action_dim), device=h_s.device)
        for m in self.modules:
            out = m(h_s, prev_outputs)
            prev_outputs = torch.cat([prev_outputs, out.unsqueeze(1)], dim=1)
        # The last output corresponds to the current task
        return out