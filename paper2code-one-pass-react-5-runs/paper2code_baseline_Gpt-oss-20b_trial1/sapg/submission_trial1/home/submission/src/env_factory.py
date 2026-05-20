"""
Utility for creating a vectorised gymnasium environment.
"""

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np

def make_vec_env(env_id: str, num_envs: int, seed: int = 0):
    """
    Create a vectorised environment with `num_envs` parallel workers.

    Parameters
    ----------
    env_id : str
        Gym environment id.
    num_envs : int
        Number of parallel workers.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    gymnasium.vector.AsyncVectorEnv
        Vectorised environment.
    """
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id, render_mode=None)
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
            return env
        return _thunk

    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = AsyncVectorEnv(env_fns)
    return vec_env