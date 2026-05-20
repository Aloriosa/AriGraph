"""
A small continuous‑action wrapper around CartPole-v1 to match the
action space used in the paper (Gaussian policy).
The original CartPole has a discrete action space; we convert it to
a 1‑dimensional continuous action between -1.0 and 1.0.
"""

import gymnasium as gym
import numpy as np

class CartPoleContinuous(gym.Env):
    """
    Continuous action version of CartPole.
    Action: 1‑dim continuous value in [-1, 1].
    The action is mapped to either 0 or 1 (left/right) by thresholding at 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = self.env.observation_space
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        # Map continuous action to discrete 0/1
        discrete_action = 0 if action[0] < 0 else 1
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        done = terminated or truncated
        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()