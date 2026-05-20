import gymnasium as gym
import random
from typing import List, Tuple
import numpy as np

class MixedResetEnv(gym.Env):
    """
    Wraps a gymnasium environment to provide a mixed initial state distribution:
    With probability p, the episode starts from a randomly chosen critical state,
    otherwise a standard environment reset is performed.
    """
    def __init__(self, env: gym.Env, critical_states: np.ndarray, p: float):
        super().__init__()
        self.env = env
        self.critical_states = critical_states
        self.p = p
        # expose the same observation/action specs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        if random.random() < self.p and len(self.critical_states) > 0:
            # choose a critical state and set it
            state = random.choice(self.critical_states)
            # many gym environments expose the underlying state as `state`
            if hasattr(self.env, "state"):
                self.env.state = state.copy()
            else:
                # fallback: try to reset and then set the observation
                # (not guaranteed to work on all envs)
                _, _ = self.env.reset()
                self.env.state = state.copy()
        else:
            self.env.reset(seed=seed, options=options)
        # return the current observation; for environments without explicit reset obs,
        # we just return the state
        obs = getattr(self.env, "state", self.env.reset()[0])
        return obs, {}

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()