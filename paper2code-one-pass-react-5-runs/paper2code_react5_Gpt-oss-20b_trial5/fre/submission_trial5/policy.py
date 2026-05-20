import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    """Gaussian policy conditioned on state and latent z."""
    def __init__(self, state_dim: int, latent_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2)  # mean & log_std
        )

    def forward(self, state: torch.Tensor, z: torch.Tensor):
        x = torch.cat([state, z], dim=-1)
        out = self.net(x)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std


class QNet(nn.Module):
    """Q‑function conditioned on state, action and latent z."""
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, z: torch.Tensor):
        x = torch.cat([state, action, z], dim=-1)
        return self.net(x).squeeze(-1)