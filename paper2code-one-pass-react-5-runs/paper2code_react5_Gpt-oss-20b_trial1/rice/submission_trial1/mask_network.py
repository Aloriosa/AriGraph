import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import Flatten
from gymnasium import spaces

class MaskNet(nn.Module):
    """
    Simple MLP that outputs probability of NOT masking (importance score).
    Output in (0,1) via sigmoid.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 256, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)  # shape (batch,)

def train_mask_network(env, policy, mask_net: MaskNet, device, timesteps: int, alpha: float, batch_size: int = 2048):
    """
    Train mask network to maximize total reward + alpha * mask probability.
    Uses a simple PPO-like update with clipped surrogate objective.
    """
    mask_net.train()
    optimizer = optim.Adam(mask_net.parameters(), lr=3e-4)
    gamma = 0.99
    eps_clip = 0.2

    obs = torch.tensor(env.reset(seed=42)[0], dtype=torch.float32, device=device)
    data = []

    for step in range(timesteps):
        with torch.no_grad():
            action, _ = policy.predict(obs.cpu().numpy(), deterministic=True)
        action = action.reshape(1, -1)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Compute mask probability for current observation
        mask_prob = mask_net(torch.tensor(obs, dtype=torch.float32, device=device))
        # Intrinsic reward for mask
        intrinsic = alpha * mask_prob

        total_reward = reward + intrinsic.item()

        data.append((obs, action, reward, mask_prob, done))

        if len(data) >= batch_size or done:
            # Compute returns and advantages
            R = 0 if done else env.get_normalized_observation(next_obs)[0]
            returns = []
            for _, _, r, _, d in reversed(data):
                R = r + gamma * R * (1 - d)
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)

            obs_batch = torch.stack([d[0] for d in data])
            action_batch = torch.cat([torch.tensor(d[1], dtype=torch.float32, device=device) for d in data])
            mask_batch = torch.stack([d[3] for d in data])

            # Compute advantage: total reward - baseline (policy's value estimate)
            with torch.no_grad():
                _, values = policy.predict(obs_batch.cpu().numpy(), deterministic=True, return_dict=True)
                values = torch.tensor(values, dtype=torch.float32, device=device)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss (clipped surrogate)
            probs_old = torch.tensor([policy.predict(o.cpu().numpy(), deterministic=True)[0] for o in obs_batch], dtype=torch.float32, device=device)
            probs_new = torch.tensor([policy.predict(o.cpu().numpy(), deterministic=True)[0] for o in obs_batch], dtype=torch.float32, device=device)
            ratio = probs_new / (probs_old + 1e-8)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Mask loss (maximize mask probability)
            mask_loss = -mask_batch.mean()

            loss = policy_loss + mask_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data = []

        if done:
            obs = torch.tensor(env.reset(seed=42)[0], dtype=torch.float32, device=device)
        else:
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    mask_net.eval()