import torch
import torch.nn as nn

class FREDecoder(nn.Module):
    """
    Simple MLP that predicts reward for a single state given the latent z.
    """
    def __init__(self, state_dim=10, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, z):
        """
        state: (B, state_dim)
        z: (B, latent_dim)
        """
        x = torch.cat([state, z], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)