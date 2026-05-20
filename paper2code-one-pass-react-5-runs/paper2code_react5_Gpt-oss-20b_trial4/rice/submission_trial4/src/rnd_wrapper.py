import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


class RNDWrapper(gym.Wrapper):
    """
    Wraps an env to add a Random Network Distillation (RND) exploration bonus.
    Also supports a mixed initial state distribution:
      with probability `p`, the env starts from a randomly chosen critical state.
      otherwise it starts from the default initial state.
    """

    def __init__(self, env, lambda_rnd=0.01, critical_states=None, p=0.25):
        super().__init__(env)
        self.lambda_rnd = lambda_rnd
        self.critical_states = critical_states or []
        self.p = p

        obs_dim = env.observation_space.shape[0]
        # Target network (fixed random weights)
        self.target_net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        for param in self.target_net.parameters():
            param.requires_grad = False
        # Predictor network (trained)
        self.predictor_net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.predictor_optimizer = optim.Adam(self.predictor_net.parameters(), lr=1e-3)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Compute RND bonus
        with torch.no_grad():
            target_feat = self.target_net(torch.FloatTensor(obs))
        pred_feat = self.predictor_net(torch.FloatTensor(obs))
        bonus = torch.mean((pred_feat - target_feat) ** 2).item()

        # Train predictor
        loss = torch.mean((pred_feat - target_feat) ** 2)
        self.predictor_optimizer.zero_grad()
        loss.backward()
        self.predictor_optimizer.step()

        # Augment reward
        augmented_reward = reward + self.lambda_rnd * bonus
        return obs, augmented_reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        if self.p > 0 and self.critical_states and random.random() < self.p:
            # Start from a critical state
            state = random.choice(self.critical_states)
            # For envs that expose `state`, set it directly
            if hasattr(self.env, "state"):
                self.env.state = state
            # Return the state as observation
            return state, {}
        else:
            return self.env.reset(*args, **kwargs)