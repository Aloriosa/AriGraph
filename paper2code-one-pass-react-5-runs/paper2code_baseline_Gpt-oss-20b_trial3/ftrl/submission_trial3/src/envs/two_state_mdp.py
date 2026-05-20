"""
Very small two‑state MDP used for analytical sanity checks.
State 0 -> state 1 with probability 1, reward r0
State 1 -> state 0 with probability 1-f_theta, reward r1
Policy selects f_theta in [0,1] (probability to stay in state 1).
"""
import numpy as np

class TwoStateMDP:
    def __init__(self, r0=1.0, r1=1.0, gamma=0.99):
        self.r0 = r0
        self.r1 = r1
        self.gamma = gamma

    def step(self, f_theta):
        # f_theta = prob to stay in state 1
        # we only need to sample next state for simulation
        pass