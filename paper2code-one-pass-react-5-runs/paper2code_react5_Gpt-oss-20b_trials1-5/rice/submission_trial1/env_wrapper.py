import gymnasium as gym
import numpy as np
import torch
from rnd import RNDModule
from utils import set_seed

class RICEEnvWrapper(gym.Wrapper):
    """
    Wrapper that adds RND intrinsic reward to the environment.
    """
    def __init__(self, env, rnd: RNDModule, lambda_: float):
        super().__init__(env)
        self.rnd = rnd
        self.lambda_ = lambda_
        self.obs_dim = env.observation_space.shape[0]
        self.rnd_device = self.rnd.device
        self.reset()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Compute intrinsic reward
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.rnd_device)
        intrinsic = self.rnd.compute_bonus(obs_tensor).item()
        # Update predictor
        self.rnd.update(obs_tensor.unsqueeze(0))
        reward += self.lambda_ * intrinsic
        self.last_obs = obs
        return obs, reward, terminated, truncated, info