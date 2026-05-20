import d4rl.qlearning_dataset as qlearn
import torch

class OfflineDataset:
    """
    Wrapper around D4RL's offline datasets.
    """
    def __init__(self, env_name: str):
        data = qlearn.load_dataset(env_name)
        self.data = data
        self.observations = torch.tensor(data['observations'], dtype=torch.float32)
        self.next_observations = torch.tensor(data['next_observations'], dtype=torch.float32)
        self.actions = torch.tensor(data['actions'], dtype=torch.float32)
        self.rewards = torch.tensor(data['rewards'], dtype=torch.float32)
        self.dones = torch.tensor(data['terminals'], dtype=torch.float32)
        self.length = len(self.observations)

    def sample_batch(self, batch_size: int):
        idx = torch.randint(0, self.length, (batch_size,))
        return dict(
            observations=self.observations[idx],
            next_observations=self.next_observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx]
        )