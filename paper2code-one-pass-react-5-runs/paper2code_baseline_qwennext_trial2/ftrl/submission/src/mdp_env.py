import numpy as np

class TwoStateMDP:
    """
    Two‑state MDP with deterministic transitions:
    - States: 0 (s0), 1 (s1)
    - Actions: 0 (stay), 1 (move)
    - Transition:
        * From s0, action 1 -> s1, reward 1
        * From s1, action 0 -> s0, reward 0
    The episode ends after one step (since we only care about reaching s1).
    """
    def __init__(self):
        self.state = None

    def reset(self, start_state=0):
        self.state = start_state
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        if self.state == 0:
            if action == 1:
                next_state = 1
                reward = 1.0
            else:
                next_state = 0
                reward = 0.0
        else:  # state == 1
            if action == 0:
                next_state = 0
                reward = 0.0
            else:
                next_state = 1
                reward = 0.0
        self.state = next_state
        done = True  # one‑step episode
        return np.array([self.state], dtype=np.float32), reward, done, {}