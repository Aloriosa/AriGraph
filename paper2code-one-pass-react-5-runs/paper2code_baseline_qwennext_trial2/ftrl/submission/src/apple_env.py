import numpy as np

class AppleRetrieval:
    """
    1‑D gridworld.
    Phase 1: start at 0, goal at M (go right).
    Phase 2: start at M, goal at 0 (go left).
    Observation: sign indicating phase:
        * -1 for Phase 1
        * +1 for Phase 2
    Action: 0 (left), 1 (right)
    Episode ends when goal reached or after 100 steps.
    Reward: +1 for each correct step, -1 for incorrect step.
    """
    def __init__(self, M=10):
        self.M = M
        self.pos = None
        self.phase = None  # 1 or 2

    def reset(self, phase=1):
        self.phase = phase
        if phase == 1:
            self.pos = 0
        else:
            self.pos = self.M
        obs = np.array([self.phase * -1], dtype=np.float32)  # -1 for phase 1, +1 for phase 2
        return obs

    def step(self, action):
        # action: 0 left, 1 right
        reward = 0.0
        if self.phase == 1:
            if action == 1:
                self.pos += 1
            else:
                self.pos -= 1
            if self.pos == self.M:
                reward = 1.0
        else:  # phase 2
            if action == 0:
                self.pos -= 1
            else:
                self.pos += 1
            if self.pos == 0:
                reward = 1.0

        # Clip position to [0, M]
        self.pos = max(0, min(self.M, self.pos))
        done = (self.pos == self.M and self.phase == 1) or (self.pos == 0 and self.phase == 2)
        obs = np.array([self.phase * -1], dtype=np.float32)
        return obs, reward, done, {}