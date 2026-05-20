#!/usr/bin/env python3
"""
Utility helpers for reward functions, data handling, and training.
"""
import numpy as np
import random
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Callable, List, Tuple, Dict

# ---------- Reward function families ----------
class RewardFunction:
    """Abstract base for a reward function over states."""
    def __call__(self, state: np.ndarray) -> float:
        raise NotImplementedError

class GoalReachReward(RewardFunction):
    """Reward of -1 until goal state is reached within tolerance."""
    def __init__(self, goal_state: np.ndarray, tolerance: float = 0.1):
        self.goal_state = goal_state
        self.tol = tolerance

    def __call__(self, state: np.ndarray) -> float:
        return 0.0 if np.linalg.norm(state - self.goal_state) < self.tol else -1.0

class LinearReward(RewardFunction):
    """Linear reward: r = w·state."""
    def __init__(self, weight: np.ndarray):
        self.w = weight

    def __call__(self, state: np.ndarray) -> float:
        return float(np.dot(self.w, state))

class MLPReward(RewardFunction):
    """Small 2‑layer MLP reward."""
    def __init__(self, in_dim: int, hidden: int = 32):
        self.w1 = np.random.randn(in_dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.w2 = np.random.randn(hidden, 1) * 0.1
        self.b2 = np.zeros(1)

    def __call__(self, state: np.ndarray) -> float:
        h = np.tanh(state @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return float(np.clip(out, -1.0, 1.0))

def sample_random_reward(state_dim: int) -> RewardFunction:
    """Sample a random reward function from the three families."""
    choice = random.choice(["goal", "lin", "mlp"])
    if choice == "goal":
        goal = np.random.uniform(-4, 4, size=state_dim)  # arbitrary range
        return GoalReachReward(goal_state=goal)
    elif choice == "lin":
        w = np.random.uniform(-1, 1, size=state_dim)
        return LinearReward(weight=w)
    else:
        return MLPReward(in_dim=state_dim)


# ---------- Dataset wrapper ----------
class OfflineDataset(Dataset):
    """Simple wrapper around the offline transitions."""
    def __init__(self, transitions: List[Dict]):
        self.trans = transitions

    def __len__(self):
        return len(self.trans)

    def __getitem__(self, idx):
        t = self.trans[idx]
        return (
            torch.tensor(t["state"], dtype=torch.float32),
            torch.tensor(t["action"], dtype=torch.long),
            torch.tensor(t["next_state"], dtype=torch.float32),
            torch.tensor(t["env_reward"], dtype=torch.float32),
        )