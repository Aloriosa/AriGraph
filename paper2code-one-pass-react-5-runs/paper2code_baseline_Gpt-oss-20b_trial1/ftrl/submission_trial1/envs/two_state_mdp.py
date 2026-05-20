import gym
from gym import spaces
import numpy as np

class TwoStateMDP(gym.Env):
    """
    A toy 2‑state MDP used to demonstrate forgetting of pre‑trained
    capabilities.  The environment has two states (0 and 1) and two
    deterministic actions (0 and 1).  The transition dynamics and
    rewards are fixed but can be changed by the user.

    Parameters
    ----------
    start_state : int, optional
        The state from which each episode starts.  If ``None`` the
        starting state is chosen uniformly at random.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, start_state=None, seed=0):
        super().__init__()
        self.observation_space = spaces.Discrete(2)  # states 0 or 1
        self.action_space = spaces.Discrete(2)      # actions 0 or 1
        self.start_state = start_state
        self.seed(seed)
        self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, **kwargs):
        if self.start_state is None:
            self.state = self.observation_space.sample()
        else:
            self.state = self.start_state
        return np.array([self.state], dtype=np.int32)

    def step(self, action):
        # deterministic transition: action 1 moves to the other state,
        # action 0 stays in the same state
        if action == 1:
            self.state = 1 - self.state
        # reward: +5 if in state 1, -1 if in state 0
        reward = 5 if self.state == 1 else -1
        done = False
        return np.array([self.state], dtype=np.int32), reward, done, {}

    def render(self, mode="human"):
        print(f"State: {self.state}")