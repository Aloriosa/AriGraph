import gym
import numpy as np
from gym.vector import SyncVectorEnv

def make_vec_env(env_name, num_envs, seed=0):
    """
    Create a vectorised gym environment with `num_envs` copies of `env_name`.
    Each env receives a different seed for reproducibility.
    """
    def _make_env(rank):
        def _thunk():
            env = gym.make(env_name)
            env.seed(seed + rank)
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
            return env
        return _thunk

    envs = SyncVectorEnv([_make_env(i) for i in range(num_envs)])
    return envs