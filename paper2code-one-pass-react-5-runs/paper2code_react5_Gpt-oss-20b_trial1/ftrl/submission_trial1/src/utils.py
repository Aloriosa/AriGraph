# src/utils.py
"""
Utility functions for computing Fisher information and other helpers.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


def compute_fisher(policy: nn.Module, env, device, num_samples=200):
    """
    Estimate diagonal Fisher information matrix for the policy
    on the given environment by sampling trajectories.
    """
    policy.eval()
    fisher = defaultdict(lambda: torch.zeros_like(next(policy.parameters())))

    for _ in range(num_samples):
        obs = torch.tensor(env.reset(), dtype=torch.float32).to(device)
        done = False
        while not done:
            logits, _ = policy.forward(obs.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            policy.zero_grad()
            (-logp).backward()
            for name, p in policy.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
            # Step env
            obs_next, _, done, _ = env.step(action.item())
            obs = torch.tensor(obs_next, dtype=torch.float32).to(device)

    # Average over number of samples
    for name in fisher:
        fisher[name] /= num_samples
    return fisher