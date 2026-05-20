"""
AppleRetrieval environment
Two‑phase 1‑D gridworld as described in the paper (Appendix A.2).
Phase 1: move from x=0 to x=M to collect an apple (reward +1 per step right,
-1 per step left).
Phase 2: move back from x=M to x=0 (reward +1 per step left,
-1 per step right).
The observation is a scalar c ∈ {−1, +1} indicating the phase.
Episode terminates when the agent reaches the goal or after 100 timesteps.
"""

import numpy as np

class AppleRetrieval:
    def __init__(self, M=30, max_steps=100, seed=None):
        self.M = M
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.phase = 1
        self.x = 0
        self.step = 0
        self.done = False
        return self._obs()

    def _obs(self):
        # observation is a scalar: -1 for phase1, +1 for phase2
        return np.array([ -1.0 if self.phase == 1 else 1.0 ], dtype=np.float32)

    def step(self, action):
        """
        Action: 0 = left, 1 = right
        """
        if self.done:
            raise RuntimeError("Episode already terminated")

        self.step += 1
        if action == 1:
            self.x += 1
        else:
            self.x -= 1

        # Clip to the grid [0, M]
        self.x = np.clip(self.x, 0, self.M)

        # Reward logic
        if self.phase == 1:
            reward = 1.0 if action == 1 else -1.0
            if self.x == self.M:
                self.phase = 2
        else:  # phase 2
            reward = 1.0 if action == 0 else -1.0
            if self.x == 0:
                self.done = True  # finished retrieving apple

        if self.step >= self.max_steps:
            self.done = True

        return self._obs(), reward, self.done, {}

    def sample_action(self, policy, obs):
        """
        Convenience: sample an action from a policy given an observation.
        """
        probs = policy(obs)
        return self.rng.choice([0, 1], p=probs)