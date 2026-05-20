import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv

def make_cartpole_vec_env(num_envs: int, seed: int = 42):
    """
    Create a vectorised CartPole-v1 environment.
    """
    def _make_env(rank):
        def init():
            env = gym.make("CartPole-v1")
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
            env.reset(seed=seed + rank)
            return env
        return init

    env_fns = [_make_env(i) for i in range(num_envs)]
    vec_env = SyncVectorEnv(env_fns)
    return vec_env