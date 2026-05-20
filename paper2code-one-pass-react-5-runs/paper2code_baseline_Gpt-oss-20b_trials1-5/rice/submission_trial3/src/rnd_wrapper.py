import numpy as np
import torch
import gymnasium as gym
from .rnd import MLP
from typing import Optional

class RNDWrapper(gym.Env):
    """
    Gym wrapper that augments the environment reward with an RND intrinsic bonus.
    It also trains the predictor network online.
    """
    def __init__(self, env: gym.Env, lambda_: float, device: Optional[str] = None):
        super().__init__()
        self.env = env
        self.lambda_ = lambda_
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Observation dimension
        obs_dim = env.observation_space.shape[0]

        # Target network (fixed random weights)
        self.target = MLP(obs_dim, hidden_dim=128, output_dim=128).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor network (trainable)
        self.predictor = MLP(obs_dim, hidden_dim=128, output_dim=128).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

        # expose same spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # compute intrinsic reward
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            target = self.target(obs_t)
        pred = self.predictor(obs_t)
        intrinsic = (self.lambda_ * ((target - pred)**2).mean().item())
        reward += intrinsic

        # train predictor
        loss = ((target - pred)**2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return obs, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()