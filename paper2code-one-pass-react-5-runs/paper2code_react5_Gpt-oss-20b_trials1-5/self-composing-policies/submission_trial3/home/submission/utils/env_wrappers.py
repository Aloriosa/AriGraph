import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SequentialTaskWrapper(gym.Wrapper):
    """
    Wraps a list of environments into a continuous sequence.
    Each task is run for `max_steps_per_task` timesteps.
    """

    def __init__(self, env_list, max_steps_per_task=1_000_000):
        super().__init__(env_list[0])
        self.envs = env_list
        self.max_steps_per_task = max_steps_per_task
        self.current_task = 0
        self.step_counter = 0

    def reset(self, **kwargs):
        obs, info = self.envs[self.current_task].reset(**kwargs)
        self.step_counter = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.envs[
            self.current_task
        ].step(action)
        self.step_counter += 1
        if self.step_counter >= self.max_steps_per_task or terminated or truncated:
            # Move to next task
            self.current_task += 1
            if self.current_task >= len(self.envs):
                # End of all tasks
                done = True
                self.current_task = len(self.envs) - 1  # stay at last
            else:
                # Reset new env
                obs, info = self.envs[self.current_task].reset()
                self.step_counter = 0
            done = True
        else:
            done = False
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.envs[self.current_task].render(mode=mode)

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space