"""
Synthetic offline dataset generator.
Creates a buffer of (state, action, next_state) tuples.
"""

import numpy as np

class OfflineDataset:
    def __init__(self, state_dim=2, action_dim=2, num_trajectories=200, traj_len=50, seed=0):
        rng = np.random.default_rng(seed)
        self.states = []
        self.actions = []
        self.next_states = []

        for _ in range(num_trajectories):
            state = rng.normal(size=state_dim)
            for _ in range(traj_len):
                action = rng.normal(size=action_dim)
                # Simple linear dynamics with noise
                next_state = state + action + rng.normal(scale=0.1, size=state_dim)
                self.states.append(state)
                self.actions.append(action)
                self.next_states.append(next_state)
                state = next_state

        self.states = np.array(self.states)          # shape (N, state_dim)
        self.actions = np.array(self.actions)        # shape (N, action_dim)
        self.next_states = np.array(self.next_states)  # shape (N, state_dim)

    def sample_batch(self, batch_size, seed=None):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.states), size=batch_size, replace=False)
        return {
            'states': self.states[idx],
            'actions': self.actions[idx],
            'next_states': self.next_states[idx]
        }

    def num_samples(self):
        return len(self.states)