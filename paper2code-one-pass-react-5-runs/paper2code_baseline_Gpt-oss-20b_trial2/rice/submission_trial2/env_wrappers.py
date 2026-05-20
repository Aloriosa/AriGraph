import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------- RND ------------------------------------
class RND:
    """Random Network Distillation (RND) module."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, device='cpu'):
        self.device = device
        # Random target network (fixed)
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        # Predictor network (trainable)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def compute_intrinsic(self, obs: np.ndarray) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        target = self.target(obs_t)
        pred = self.predictor(obs_t)
        return ((target - pred) ** 2).mean().item()

    def update(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        target = self.target(obs_t).detach()
        pred = self.predictor(obs_t)
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RNDWrapper(gym.Wrapper):
    """Adds RND intrinsic reward to the environment."""
    def __init__(self, env: gym.Env, rnd: RND, intrinsic_scale: float = 0.01, device='cpu'):
        super().__init__(env)
        self.rnd = rnd
        self.scale = intrinsic_scale
        self.device = device

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        intrinsic = self.rnd.compute_intrinsic(obs)
        self.rnd.update(obs)
        reward += self.scale * intrinsic
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ---------- Mixed Initial Distribution -------------------
class MixedInitWrapper(gym.Wrapper):
    """
    With probability `p`, reset to a randomly chosen critical state.
    Otherwise, perform a normal reset.
    """
    def __init__(self, env: gym.Env, critical_states: list, p: float = 0.25, seed: int = 42):
        super().__init__(env)
        self.critical_states = critical_states
        self.p = p
        self.rng = np.random.RandomState(seed)

    def reset(self, **kwargs):
        if self.rng.rand() < self.p and self.critical_states:
            state = self.rng.choice(self.critical_states)
            # CartPole has a `state` attribute that holds the observation
            self.env.state = state
            # Return the observation (CartPole expects a float32 array)
            return np.array(state, dtype=np.float32)
        else:
            return self.env.reset(**kwargs)