import numpy as np
from typing import Callable, Tuple

def random_goal_reaching(state_dim: int, rng: np.random.Generator) -> Tuple[Callable, np.ndarray]:
    """Return a reward function that gives -1 until a random goal is reached."""
    goal = rng.uniform(-1.0, 1.0, size=(state_dim,))
    def reward_fn(state: np.ndarray) -> float:
        # Euclidean distance to goal
        dist = np.linalg.norm(state - goal)
        return -1.0 if dist > 0.05 else 0.0
    return reward_fn, goal

def random_mlp(state_dim: int, rng: np.random.Generator) -> Tuple[Callable, Tuple]:
    """Return a 2‑layer MLP reward function."""
    hidden_dim = 32
    w1 = rng.normal(0, 1.0, size=(hidden_dim, state_dim))
    b1 = rng.normal(0, 1.0, size=(hidden_dim,))
    w2 = rng.normal(0, 1.0, size=(1, hidden_dim))
    b2 = rng.normal(0, 1.0, size=(1,))

    def reward_fn(state: np.ndarray) -> float:
        h = np.tanh(np.dot(w1, state) + b1)
        out = np.dot(w2, h) + b2
        return float(np.clip(out[0], -1.0, 1.0))
    return reward_fn, (w1, b1, w2, b2)

def sample_random_reward(state_dim: int, rng: np.random.Generator) -> Tuple[Callable, Tuple]:
    """Sample a random reward function from the mixture."""
    choice = rng.choice(['goal', 'mlp'])
    if choice == 'goal':
        return random_goal_reaching(state_dim, rng)
    else:
        return random_mlp(state_dim, rng)