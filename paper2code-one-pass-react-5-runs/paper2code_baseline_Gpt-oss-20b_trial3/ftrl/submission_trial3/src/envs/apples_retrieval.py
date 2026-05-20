import numpy as np

class AppleRetrieval:
    """
    1‑D gridworld toy environment.
    Phase 1: start at 0, goal at M (apple).
    Phase 2: start at M, goal at 0 (return).
    Rewards: +1 for moving in the correct direction, -1 otherwise.
    Observation: scalar c or -c depending on phase.
    """
    def __init__(self, phase=0, M=10, max_steps=100, c=1.0):
        """
        phase: 0 = full task (both phases), 1 = Phase 1 only, 2 = Phase 2 only
        """
        self.phase = phase
        self.M = M
        self.max_steps = max_steps
        self.c = c
        self.reset()

    def reset(self):
        self.t = 0
        if self.phase == 0:
            # Full task: start at 0, but we will generate episodes with random phase
            self.phase = np.random.choice([1, 2])
        if self.phase == 1:
            self.pos = 0
            self.target = self.M
            self.obs = np.array([-self.c], dtype=np.float32)
        else:
            self.pos = self.M
            self.target = 0
            self.obs = np.array([self.c], dtype=np.float32)
        return self.obs

    def step(self, action):
        """
        action: 0 = left, 1 = right
        """
        self.t += 1
        step = 1 if action == 1 else -1
        new_pos = self.pos + step
        # clip to bounds
        new_pos = max(0, min(self.M, new_pos))
        reward = 1.0 if (new_pos - self.pos) == (self.target - self.pos) else -1.0
        self.pos = new_pos
        done = self.pos == self.target or self.t >= self.max_steps
        return np.array([self.c if self.phase == 2 else -self.c], dtype=np.float32), reward, done, {}

    def render(self):
        pass