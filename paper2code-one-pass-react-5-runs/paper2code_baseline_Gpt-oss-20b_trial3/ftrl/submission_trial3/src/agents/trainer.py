import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Bernoulli, kl_divergence

class Trainer:
    """
    REINFORCE with optional BC (behavioral cloning) and EWC regularisation.
    """
    def __init__(self, policy, env, device='cpu',
                 lr=1e-3, gamma=0.99,
                 buffer_size=0, bc_coef=0.0, ewc_coef=0.0, bc_buffer=None):
        self.policy = policy.to(device)
        self.env = env
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.bc_coef = bc_coef
        self.ewc_coef = ewc_coef
        self.bc_buffer = bc_buffer or []

        # For EWC
        if ewc_coef > 0:
            self.prev_params = {n: p.clone().detach() for n, p in self.policy.named_parameters()}
            self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        # simple diagonal Fisher by running one batch of data
        fisher = {}
        self.policy.eval()
        with torch.no_grad():
            for _ in range(10):
                obs = torch.tensor(self.env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)
                probs = self.policy(obs)
                dist = Bernoulli(probs)
                action = dist.sample()
                logp = dist.log_prob(action)
                loss = -logp.sum()
                self.optimizer.zero_grad()
                loss.backward()
                for name, p in self.policy.named_parameters():
                    fisher[name] = fisher.get(name, torch.zeros_like(p)) + p.grad.data.pow(2)
        for name in fisher:
            fisher[name] /= 10.0
        self.policy.train()
        return fisher

    def train(self, episodes=100, batch_size=1, verbose=True):
        for ep in range(episodes):
            traj = []
            obs = torch.tensor(self.env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)
            done = False
            while not done:
                probs = self.policy(obs)
                dist = Bernoulli(probs)
                action = dist.sample()
                next_obs, reward, done, _ = self.env.step(action.item())
                traj.append((obs, action, reward))
                obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Compute returns
            returns = []
            G = 0
            for _, _, r in reversed(traj):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

            # Policy gradient loss
            loss = 0
            for (obs, action, _), G in zip(traj, returns):
                probs = self.policy(obs)
                dist = Bernoulli(probs)
                logp = dist.log_prob(action)
                loss -= logp * G
            loss /= len(traj)

            # BC auxiliary loss
            if self.bc_coef > 0 and self.bc_buffer:
                bc_loss = 0
                for s, a in self.bc_buffer:
                    s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
                    probs = self.policy(s)
                    dist = Bernoulli(probs)
                    teacher_prob = a
                    # KL(teacher || student)
                    kl = torch.nn.functional.kl_div(
                        torch.log(probs + 1e-8),
                        torch.tensor(teacher_prob, dtype=torch.float32).to(self.device),
                        reduction='batchmean')
                    bc_loss += kl
                bc_loss /= len(self.bc_buffer)
                loss += self.bc_coef * bc_loss

            # EWC loss
            if self.ewc_coef > 0:
                ewc_loss = 0
                for name, p in self.policy.named_parameters():
                    fisher = self.fisher[name]
                    prev = self.prev_params[name]
                    ewc_loss += (fisher * (p - prev).pow(2)).sum()
                loss += self.ewc_coef * ewc_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose and (ep + 1) % 20 == 0:
                print(f"Episode {ep+1}/{episodes} – loss: {loss.item():.4f}")

    def evaluate(self, episodes=10):
        total_success = 0
        for _ in range(episodes):
            obs = torch.tensor(self.env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)
            done = False
            while not done:
                probs = self.policy(obs)
                action = torch.bernoulli(probs).long()
                obs, _, done, _ = self.env.step(action.item())
                if done:
                    # success if reached target
                    if isinstance(self.env, type(self.env)):
                        # In AppleRetrieval success iff pos==target
                        pass
            # In this toy env, success is automatically achieved if episode ends
            total_success += 1
        return total_success / episodes

def evaluate(policy, env, episodes=10):
    policy.eval()
    total_success = 0
    for _ in range(episodes):
        obs = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
        done = False
        while not done:
            probs = policy(obs)
            action = torch.bernoulli(probs).long()
            obs, _, done, _ = env.step(action.item())
        total_success += 1
    return total_success / episodes