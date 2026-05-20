import torch
import numpy as np

class RewardFunction:
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GoalReward(RewardFunction):
    """Sparse goal‑reaching reward: 0 at goal, -1 otherwise."""
    def __init__(self, goal: torch.Tensor, tol: float = 2.0):
        self.goal = goal
        self.tol = tol

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(states - self.goal, dim=-1)
        return torch.where(dist <= self.tol,
                           torch.zeros_like(dist),
                           -torch.ones_like(dist))


class LinearReward(RewardFunction):
    """Linear reward: r = w·s."""
    def __init__(self, weights: torch.Tensor):
        self.weights = weights

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        return states @ self.weights


class MLPReward(RewardFunction):
    """Random 2‑layer MLP."""
    def __init__(self, state_dim: int):
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


def sample_reward_function(data: dict,
                           state_dim: int,
                           mixture=[0.33, 0.33, 0.34]) -> RewardFunction:
    """
    Sample a reward function from a uniform mixture of:
    1. Goal‑reaching
    2. Linear
    3. MLP
    """
    rng = np.random.default_rng()
    choice = rng.choice(3, p=mixture)

    if choice == 0:  # goal
        obs = data['observations']
        idx = rng.integers(len(obs))
        goal = torch.tensor(obs[idx], dtype=torch.float32)
        return GoalReward(goal)
    elif choice == 1:  # linear
        w = torch.randn(state_dim)
        return LinearReward(w)
    else:  # MLP
        return MLPReward(state_dim)