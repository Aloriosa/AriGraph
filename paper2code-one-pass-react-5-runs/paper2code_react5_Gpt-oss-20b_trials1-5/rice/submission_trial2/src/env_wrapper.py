import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from .utils import MLP

class RNDWrapper(gym.Wrapper):
    """
    Adds a Random Network Distillation (RND) intrinsic reward.
    Also exposes a method to reset to an arbitrary MuJoCo state.
    """
    def __init__(self, env, rnd_hidden=128, intrinsic_coef=0.01, device='cpu'):
        super().__init__(env)
        self.device = device
        obs_dim = env.observation_space.shape[0]
        # Target network – fixed
        self.target = MLP(obs_dim, rnd_hidden, rnd_hidden).to(self.device)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False
        # Predictor network – trainable
        self.predictor = MLP(obs_dim, rnd_hidden, rnd_hidden).to(self.device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-4)
        self.intrinsic_coef = intrinsic_coef

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Compute intrinsic reward
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            target_feat = self.target(obs_t)
        pred_feat = self.predictor(obs_t)
        intrinsic = torch.mean((pred_feat - target_feat) ** 2).item()
        # Update predictor
        loss = (pred_feat - target_feat).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Add intrinsic reward
        reward += self.intrinsic_coef * intrinsic
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # Options may contain a state to reset to
        if options and 'state' in options:
            state = options['state']  # tuple of (qpos, qvel)
            self.env.env.set_state(state)
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset(seed=seed)
        return obs, info

    def reset_to_state(self, state):
        """Convenience method to reset to a specific MuJoCo state."""
        self.env.env.set_state(state)
        return self.reset(options={'state': state})[0]