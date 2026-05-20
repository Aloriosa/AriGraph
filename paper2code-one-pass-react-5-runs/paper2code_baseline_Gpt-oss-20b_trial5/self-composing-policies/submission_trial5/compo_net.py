import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfComposingModule(nn.Module):
    """
    A single self‑composing policy module.
    Implements:
      - Output attention head (aggregates previous modules' outputs)
      - Input attention head (aggregates output head + previous outputs)
      - Internal policy (MLP) that adjusts the final output
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Output attention head
        self.W_out_Q = nn.Linear(state_dim, hidden_dim, bias=False)
        self.W_out_K = nn.Linear(action_dim, hidden_dim, bias=False)

        # Input attention head
        self.W_in_Q = nn.Linear(state_dim, hidden_dim, bias=False)
        self.W_in_K = nn.Linear(action_dim, hidden_dim, bias=False)
        self.W_in_V = nn.Linear(action_dim, hidden_dim, bias=False)

        # Internal policy (two‑layer MLP)
        self.ff = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor,
                prev_outputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor
            Batch of states, shape (B, state_dim).
        prev_outputs : torch.Tensor
            Batch of previous modules' outputs, shape (B, N_prev, action_dim).
            If N_prev == 0, this tensor is empty.

        Returns
        -------
        probs : torch.Tensor
            Action probability distribution, shape (B, action_dim).
        """
        B = state.shape[0]

        if prev_outputs.shape[1] == 0:
            # No previous modules – return a uniform distribution
            v = torch.zeros(B, self.action_dim, device=state.device)
            inp_att = torch.zeros(B, self.hidden_dim, device=state.device)
        else:
            # ---------- Output attention head ----------
            q_out = self.W_out_Q(state)                     # (B, H)
            K_out = self.W_out_K(prev_outputs.transpose(1, 2))  # (B, H, N)
            att_out = F.softmax(q_out.unsqueeze(2) @ K_out
                                 / math.sqrt(self.hidden_dim), dim=2)
            v = (att_out @ prev_outputs).squeeze(1)          # (B, A)

            # ---------- Input attention head ----------
            # Concatenate previous outputs with the output head
            combined = torch.cat([prev_outputs, v.unsqueeze(1)], dim=1)  # (B, N+1, A)
            K_in = self.W_in_K(combined.transpose(1, 2))                # (B, H, N+1)
            V_in = self.W_in_V(combined.transpose(1, 2))                # (B, H, N+1)
            q_in = self.W_in_Q(state)                                   # (B, H)
            att_in = F.softmax(q_in.unsqueeze(2) @ K_in
                                / math.sqrt(self.hidden_dim), dim=2)
            inp_att = (att_in @ V_in.transpose(1, 2)).squeeze(1)        # (B, H)

        # ---------- Internal policy ----------
        ff_input = torch.cat([inp_att, state], dim=1)  # (B, H+S)
        int_out = self.ff(ff_input)                   # (B, A)

        # Final action logits
        logits = v + int_out
        return F.softmax(logits, dim=1)


class CompoNet(nn.Module):
    """
    Growing network that keeps a list of SelfComposingModules.
    Each new task adds a new module and freezes all previous ones.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.modules_list = []

    def add_task(self):
        """Freeze previous modules and add a new one."""
        for m in self.modules_list:
            for p in m.parameters():
                p.requires_grad = False
        new_module = SelfComposingModule(self.state_dim,
                                        self.action_dim,
                                        self.hidden_dim)
        self.modules_list.append(new_module)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over all modules.
        Returns the action distribution of the last module.
        """
        prev_outputs = []
        for m in self.modules_list:
            if len(prev_outputs) == 0:
                prev_out_tensor = torch.empty(state.shape[0], 0,
                                              self.action_dim,
                                              device=state.device)
            else:
                prev_out_tensor = torch.stack(prev_outputs, dim=1)
            out = m(state, prev_out_tensor)  # (B, A)
            prev_outputs.append(out)
        return prev_outputs[-1]  # distribution of the last module

    def act(self, state: torch.Tensor) -> int:
        """Sample an action from the final module."""
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()