#!/usr/bin/env python3
"""
AppleRetrieval environment.

The agent starts at position 0 and must reach the apple at position M (phase 1).
After reaching the apple it must return to 0 (phase 2).

Observation:
    A single scalar value: bias +c in phase 1, -c in phase 2.

Action:
    0 = left, 1 = right

Reward:
    +1 for moving in the correct direction, -1 otherwise.
Episode terminates when the agent returns to 0 in phase 2 or after max_steps.
"""

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class AppleRetrieval(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, M=30, c=1.0, max_steps=100, seed=None):
        super().__init__()
        self.M = M          # distance to apple
        self.c = c          # observation bias
        self.max_steps = max_steps

        # Observation: single scalar bias indicating phase
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        # Actions: 0 = left, 1 = right
        self.action_space = spaces.Discrete(2)

        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.x = 0
        self.done = False
        self.steps = 0
        self.phase = 1  # 1: go to apple, 2: return
        return self._get_obs(), {}

    def _get_obs(self):
        # bias: +c in phase 1, -c in phase 2 (so that the agent can infer phase)
        bias = self.c if self.phase == 1 else -self.c
        return np.array([bias], dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        reward = 0.0

        if self.phase == 1:
            # Phase 1: move towards apple
            if action == 1:  # right
                self.x += 1
                reward = 1.0
            else:  # left
                reward = -1.0

            if self.x >= self.M:
                self.phase = 2
                self.x = self.M  # clamp at apple
        else:
            # Phase 2: return to start
            if action == 0:  # left
                self.x -= 1
                reward = 1.0
            else:  # right
                reward = -1.0

            if self.x <= 0:
                self.done = True
                self.x = 0

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def render(self, mode="human"):
        pass  # not needed for reproducibility