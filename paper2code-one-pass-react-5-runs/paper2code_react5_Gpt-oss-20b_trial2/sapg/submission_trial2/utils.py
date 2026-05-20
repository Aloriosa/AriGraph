import numpy as np
import torch

def discount_cumsum(x, discount):
    """Compute discounted cumulative sums of vectors."""
    return np.array([np.sum(x[i:] * discount ** np.arange(len(x) - i)) for i in range(len(x))])

def compute_gae(rewards, values, dones, next_value, gamma=0.99, tau=0.95):
    """
    Generalized Advantage Estimation (GAE).
    rewards: [T]
    values: [T]
    dones: [T]
    next_value: scalar (value of last state)
    Returns:
        advantages: [T]
        returns: [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * (1 - dones[t]) * next_value - values[t]
        gae = delta + gamma * tau * (1 - dones[t]) * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns

def create_dataloader(states, actions, log_probs, returns, advs, batch_size, shuffle=True):
    """Create a DataLoader for PPO minibatch updates."""
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(states).float(),
        torch.from_numpy(actions).float(),
        torch.from_numpy(log_probs).float(),
        torch.from_numpy(returns).float(),
        torch.from_numpy(advs).float()
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)