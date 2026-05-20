# src/env.py
"""
AppleRetrieval – a simple 1‑D grid world with two phases.

Phase 1: 0 → door (pos=5) → apple (pos=10)
Phase 2: 0 → apple (pos=10)

Observations are 2‑dim vectors:
    [phase_flag, normalized_position]
    phase_flag = 0.0 for Phase 1, 1.0 for Phase 2
    normalized_position = position / max_position
Actions: 0 = move left, 1 = move right
Reward: +1 for moving towards the goal, -1 for moving away
Episode ends when the apple is reached or max_steps exceeded.
"""

import numpy as np
import torch


class AppleRetrievalEnv:
    def __init__(self, phase="phase1", max_steps=100):
        assert phase in ("phase1", "phase2")
        self.phase = phase
        self.phase_flag = 0.0 if phase == "phase1" else 1.0
        self.max_steps = max_steps

        # Define phase‑specific parameters
        if phase == "phase1":
            self.door_pos = 5
            self.apple_pos = 10
        else:
            self.door_pos = None
            self.apple_pos = 10

        self.reset()

    def reset(self):
        self.pos = 0
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        # Normalized position
        max_pos = self.apple_pos
        return np.array([self.phase_flag, self.pos / max_pos], dtype=np.float32)

    def step(self, action):
        """
        action: 0 (left) or 1 (right)
        """
        assert action in (0, 1)
        self.steps += 1

        # Apply action
        if action == 1:
            self.pos += 1
        else:
            self.pos -= 1

        # Clamp position
        self.pos = max(0, min(self.pos, self.apple_pos))

        # Determine reward
        if self.phase == "phase1":
            # Reward for moving towards door or apple
            reward = 1 if action == 1 else -1
        else:
            # Phase 2: only apple matters
            reward = 1 if action == 1 else -1

        # Check terminal
        done = False
        if self.phase == "phase1":
            # Success if reached apple
            done = self.pos == self.apple_pos
        else:
            done = self.pos == self.apple_pos

        # Extra: if max steps reached, terminate
        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self):
        pass  # Not needed for this toy environment