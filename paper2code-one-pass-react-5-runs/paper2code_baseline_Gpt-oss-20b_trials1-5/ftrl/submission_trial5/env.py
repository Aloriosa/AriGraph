"""
Toy two‑phase grid‑world environment.
Phase 0 (Close): start at position 0, goal at M.
Phase 1 (Far): start at position M, goal at 0.
Rewards: +1 for moving toward goal, -1 otherwise.
Episode ends when goal reached or after 2*M steps.
"""
import numpy as np


class TwoPhaseGridWorld:
    def __init__(self, M: int = 10, phase: int = 0, max_steps: int = None, seed: int = None):
        """
        :param M: length of the grid
        :param phase: 0 (Close) or 1 (Far)
        :param max_steps: optional maximum steps per episode
        :param seed: optional RNG seed
        """
        self.M = M
        self.phase = phase
        self.max_steps = max_steps or 2 * M
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset environment to the start of the specified phase."""
        self.pos = 0 if self.phase == 0 else self.M
        self.steps = 0
        return self._state()

    def _state(self):
        """Return observation: [position, phase]"""
        return np.array([self.pos, self.phase], dtype=np.float32)

    def step(self, action: int):
        """
        :param action: 0 = left, 1 = right
        :return: state, reward, done, info
        """
        assert action in (0, 1)
        old_pos = self.pos
        if action == 0:
            self.pos = max(0, self.pos - 1)
        else:
            self.pos = min(self.M, self.pos + 1)

        # reward: +1 towards goal, -1 away
        goal_pos = self.M if self.phase == 0 else 0
        reward = 1.0 if (self.pos - old_pos) == (goal_pos - old_pos) else -1.0

        self.steps += 1
        done = (self.pos == goal_pos) or (self.steps >= self.max_steps)
        return self._state(), reward, done, {}

    def render(self):
        """Optional simple rendering."""
        print("Pos:", self.pos, "Goal:", self.M if self.phase == 0 else 0)