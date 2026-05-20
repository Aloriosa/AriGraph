import torch
import torch.nn as nn

class RNDModule:
    """
    Random Network Distillation (RND) module.
    f_target: fixed random network
    f_predictor: trainable network
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 256, device: torch.device = torch.device('cpu')):
        self.device = device
        self.f_target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        self.f_target.eval()  # fixed random weights
        self.f_predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.f_predictor.parameters(), lr=1e-4)

    def compute_bonus(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward bonus: squared L2 distance between predictor and target.
        obs: Tensor of shape (batch, obs_dim)
        """
        with torch.no_grad():
            target = self.f_target(obs)
        pred = self.f_predictor(obs)
        return ((pred - target) ** 2).mean(dim=1)

    def update(self, obs: torch.Tensor):
        """Train predictor network to match target."""
        target = self.f_target(obs).detach()
        pred = self.f_predictor(obs)
        loss = ((pred - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()