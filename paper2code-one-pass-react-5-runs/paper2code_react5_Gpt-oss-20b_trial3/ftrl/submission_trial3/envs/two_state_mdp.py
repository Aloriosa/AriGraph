import gymnasium as gym
import numpy as np

class TwoStateMDP(gym.Env):
    """
    Toy two‑state MDP from the paper.
    State 0: s0
    State 1: s1
    Transition probabilities and rewards are fixed.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, gamma=0.99, render_mode=None):
        super().__init__()
        self.gamma = gamma
        self.render_mode = render_mode
        self.state = 0

        # Observation is just the state index
        self.observation_space = gym.spaces.Discrete(2)
        # Two actions: 0 or 1
        self.action_space = gym.spaces.Discrete(2)

        # Transition table: (next_state, reward)
        self.trans = {
            0: {0: (1, 1.0), 1: (0, 0.0)},
            1: {0: (0, -1.0), 1: (1, 2.0)},
        }

    def reset(self, seed=None, options=None):
        self.state = 0
        return self.state, {}

    def step(self, action):
        next_state, reward = self.trans[self.state][action]
        self.state = next_state
        done = False  # infinite horizon
        return self.state, reward, done, False, {}

    def render(self, mode="human"):
        print(f"State: {self.state}")