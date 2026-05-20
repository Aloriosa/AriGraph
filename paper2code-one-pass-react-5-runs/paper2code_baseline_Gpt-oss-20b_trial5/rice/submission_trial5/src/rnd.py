import torch
import torch.nn as nn
import torch.nn.functional as F


class RNDTarget(nn.Module):
    """Fixed target network for RND."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class RNDPredictor(nn.Module):
    """Learned predictor network for RND."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


def rnd_bonus(predictor, target, state, device='cpu', normalize=True):
    """
    Compute the RND intrinsic reward for a given state.
    """
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        target_out = target(state)
    pred_out = predictor(state)
    bonus = (target_out - pred_out).pow(2).mean().item()
    if normalize:
        bonus = bonus / (state.shape[1] + 1e-8)
    return bonus