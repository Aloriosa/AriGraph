import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class PPO:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 lr=3e-4,
                 gamma=0.99,
                 clip=0.2,
                 epochs=4,
                 batch_size=64,
                 vf_coef=0.5,
                 ent_coef=0.01,
                 device="cpu"):
        self.policy = PolicyNet(obs_dim, act_dim).to(device)
        self.value = ValueNet(obs_dim).to(device)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.device = device

    def collect_trajectories(self, env, steps, eps=0.1):
        obs = env.reset()[0]
        obs_buf, act_buf, rew_buf, logp_buf, val_buf, done_buf = [], [], [], [], [], []
        for _ in range(steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
            logits = self.policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            val = self.value(obs_t)
            next_obs, rew, done, _, _ = env.step(action.item())
            obs_buf.append(obs)
            act_buf.append(action.item())
            rew_buf.append(rew)
            logp_buf.append(logp.item())
            val_buf.append(val.item())
            done_buf.append(done)
            if done:
                obs = env.reset()[0]
            else:
                obs = next_obs
        return (np.array(obs_buf),
                np.array(act_buf),
                np.array(rew_buf),
                np.array(logp_buf),
                np.array(val_buf),
                np.array(done_buf))

    def compute_gae(self, rewards, values, dones, gamma=None, lam=0.95):
        if gamma is None:
            gamma = self.gamma
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self,
               obs,
               actions,
               logp_old,
               returns,
               advantages,
               wrapper_losses=0.0):
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(logp_old, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.batch_size,
                                             shuffle=True)
        for _ in range(self.epochs):
            for batch in loader:
                b_obs, b_act, b_logp_old, b_ret, b_adv = batch
                logits = self.policy(b_obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(b_act)
                ratio = torch.exp(new_logp - b_logp_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                values = self.value(b_obs)
                value_loss = ((values - b_ret) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy + wrapper_losses
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()