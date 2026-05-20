"""
Utility functions for data collection from batched environments.
"""

import numpy as np
import torch
from collections import defaultdict


def collect_trajectories(agent, envs, horizon, policy_idx):
    """
    Collect a rollout of length `horizon` for each environment in `envs`
    using the policy with index `policy_idx`. Returns a dictionary of
    trajectories including:
        - states, actions, rewards, log_probs (under the policy that generated them),
          values (value estimate of the policy that generated them), dones
    """
    # Initial observations
    obs = np.array([env.reset()[0] for env in envs], dtype=np.float32)

    # Storage lists
    data = defaultdict(list)

    for _ in range(horizon):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            actions, logp, value = agent.act(obs_tensor, policy_idx=policy_idx, deterministic=False)
        actions_np = actions.cpu().numpy()
        logp_np = logp.cpu().numpy()
        value_np = value.cpu().numpy()

        # Step environments
        next_obs, rewards, terminated, truncated, _ = zip(*[env.step(action) for env, action in zip(envs, actions_np)])
        dones = np.logical_or(terminated, truncated)

        # Store data
        data["states"].append(obs)
        data["actions"].append(actions_np)
        data["rewards"].append(rewards)
        data["log_probs"].append(logp_np)
        data["values"].append(value_np)
        data["dones"].append(dones)

        obs = np.array(next_obs, dtype=np.float32)

    # Convert to numpy arrays
    for k in data:
        data[k] = np.concatenate(data[k], axis=0)

    return data