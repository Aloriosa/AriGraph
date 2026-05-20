import torch
import torch.nn as nn
import torch.nn.functional as F

class FREDecoder(nn.Module):
    """
    Decodes (state, z) -> reward.
    """
    def __init__(self, state_dim: int, latent_dim: int = 32,
                 hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = state_dim + latent_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, z: torch.Tensor):
        """
        Args:
            states: (B, 1, state_dim)
            z: (B, latent_dim)
        Returns:
            reward_pred: (B, 1)
        """
        B = states.shape[0]
        z_expanded = z.unsqueeze(1).expand(-1, states.shape[1], -1)
        inp = torch.cat([states, z_expanded], dim=-1)     # (B, 1, state+latent)
        out = self.net(inp.view(-1, inp.shape[-1]))      # (B, 1)
        return out