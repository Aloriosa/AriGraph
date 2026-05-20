import numpy as np
import torch

class SyntheticDataset:
    """
    Generates synthetic offline trajectories.
    Each trajectory is a sequence of (state, action) tuples.
    States are d‑dimensional continuous vectors.
    Actions are also continuous.
    """
    def __init__(self, num_traj=2000, traj_len=50, state_dim=10, action_dim=4, seed=42):
        np.random.seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.traj_len = traj_len

        # Random walk dynamics: next_state = state + action + noise
        self.states = []
        self.actions = []

        for _ in range(num_traj):
            s = np.random.randn(state_dim)  # init state
            traj_s = [s]
            traj_a = []
            for _ in range(traj_len - 1):
                a = np.random.randn(action_dim)
                traj_a.append(a)
                s = s + a + 0.1 * np.random.randn(state_dim)
                traj_s.append(s)
            self.states.append(np.stack(traj_s))
            self.actions.append(np.stack(traj_a))

        # Flatten into transition tuples
        self.states = np.concatenate(self.states, axis=0)  # (N, state_dim)
        self.actions = np.concatenate(self.actions, axis=0)  # (N, action_dim)

    def sample_batch(self, batch_size):
        idx = np.random.choice(len(self.states), batch_size)
        return torch.tensor(self.states[idx], dtype=torch.float32), \
               torch.tensor(self.actions[idx], dtype=torch.float32)