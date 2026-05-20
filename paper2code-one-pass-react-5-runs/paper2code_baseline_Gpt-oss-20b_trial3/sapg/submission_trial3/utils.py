import torch
from torch import nn
import numpy as np


def trw_clip(x, clip_value):
    """
    Clip values to [-clip_value, clip_value].
    """
    return torch.clamp(x, -clip_value, clip_value)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Generalised Advantage Estimation.
    """
    values = values + [0.0]  # bootstrap
    gae = 0.0
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


class RolloutBuffer:
    """
    Stores trajectory data for a single policy.
    """
    def __init__(self, obs_dim, act_dim):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []

    def add(self, obs, act, rew, done, val, logp):
        self.obs.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.dones.append(done)
        self.values.append(val)
        self.log_probs.append(logp)

    def finish_episode(self, gamma=0.99, lam=0.95):
        adv, ret = compute_gae(self.rewards, self.values, self.dones,
                               gamma=gamma, lam=lam)
        self.advantages = adv
        self.returns = ret

    def get(self):
        return (torch.tensor(self.obs, dtype=torch.float32),
                torch.tensor(self.actions, dtype=torch.float32),
                torch.tensor(self.log_probs, dtype=torch.float32),
                torch.tensor(self.returns, dtype=torch.float32),
                torch.tensor(self.advantages, dtype=torch.float32))