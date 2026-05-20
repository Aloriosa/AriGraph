import numpy as np
from typing import List, Tuple


def evaluate_policy(
    env,
    policy,
    n_episodes: int = 10,
    max_steps: int = 1_000_000,
    success_threshold: float = None,
):
    """
    Run `n_episodes` and return mean return and success rate.
    `success_threshold` is a return value; if None, success_rate is 0.0.
    """
    returns = []
    successes = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_ret = 0.0
        for _ in range(max_steps):
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total_ret += reward
            if terminated or truncated:
                break
        returns.append(total_ret)
        if success_threshold is not None:
            successes.append(1.0 if total_ret >= success_threshold else 0.0)

    mean_ret = np.mean(returns)
    success_rate = np.mean(successes) if success_threshold is not None else 0.0
    return mean_ret, success_rate