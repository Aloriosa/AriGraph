import torch
import gymnasium as gym
import numpy as np
from collections import deque
from .policy import SharedPolicy
from .utils import compute_gae, flatten_dict


class SAPGAgent:
    def __init__(self, env_name, M=3, envs_per_policy=4,
                 horizon=16, gamma=0.99, tau=0.95,
                 lr=5e-4, clip_eps=0.2, lambda_off=1.0,
                 entropy_coef=0.0, device='cpu'):
        self.M = M
        self.envs_per_policy = envs_per_policy
        self.horizon = horizon
        self.gamma = gamma
        self.tau = tau
        self.clip_eps = clip_eps
        self.lambda_off = lambda_off
        self.entropy_coef = entropy_coef
        self.device = device

        # Create environments
        self.envs = [gym.make(env_name) for _ in range(M * envs_per_policy)]
        obs_dim = self.envs[0].observation_space.shape[0]
        act_dim = self.envs[0].action_space.shape[0]

        # Shared policy backbone
        self.shared_policy = SharedPolicy(obs_dim, act_dim, hidden_dim=64,
                                         cond_dim=32).to(device)

        # Conditioning vectors φ_i for each policy
        self.conds = nn.ParameterList([
            nn.Parameter(torch.randn(32)) for _ in range(M)
        ])
        self.conds.to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.shared_policy.parameters()) + list(self.conds.parameters()),
            lr=lr)

        self.device = torch.device(device)

    def collect_data(self):
        """
        Run each policy on its block of environments for `horizon` steps.
        Returns a list of trajectory dicts per policy.
        """
        trajs = [[] for _ in range(self.M)]
        obs = [env.reset()[0] for env in self.envs]
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        for step in range(self.horizon):
            # Select policy index for each env
            policy_idx = torch.arange(len(self.envs), device=self.device) // self.envs_per_policy

            # Get conditioning vectors
            conds = torch.stack([self.conds[i] for i in policy_idx])

            # Sample actions
            mean, std = self.shared_policy.forward(obs, conds)
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()
            logp = dist.log_prob(actions).sum(-1, keepdim=True)

            # Step environments
            next_obs, rewards, dones, infos, _ = \
                zip(*[e.step(a.cpu().numpy()) for e, a in zip(self.envs, actions)])

            # Store transition
            for i, idx in enumerate(range(len(self.envs))):
                policy_id = idx // self.envs_per_policy
                trajs[policy_id].append({
                    'obs': obs[idx],
                    'act': actions[idx],
                    'logp': logp[idx],
                    'reward': torch.tensor(rewards[idx], device=self.device),
                    'done': torch.tensor(dones[idx], device=self.device),
                })

            # Prepare for next step
            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            for env, d in zip(self.envs, dones):
                if d:
                    obs[envs.index(env)] = torch.tensor(env.reset()[0],
                                                        dtype=torch.float32,
                                                        device=self.device)

        # Convert list of dicts to dict of tensors per policy
        trajs = [flatten_dict(traj) for traj in trajs]
        return trajs

    def update(self, trajs):
        """
        Perform one optimization step using on‑policy and off‑policy losses.
        """
        # Compute value estimates (simple linear value head)
        # For simplicity we use a single linear value head shared across policies
        value_head = nn.Linear(self.shared_policy.shared[-1].out_features, 1).to(self.device)
        value_head.train()

        # Compute advantage & returns for each policy
        for policy_id, traj in enumerate(trajs):
            # Dummy values: zeros (since we don't train a critic here)
            values = torch.zeros_like(traj['reward'])
            dones = traj['done']
            rewards = traj['reward']
            # Append a zero value for the bootstrap
            values = torch.cat([values, torch.tensor([0.0], device=self.device)])
            advantages, returns = compute_gae(rewards, values, dones,
                                              gamma=self.gamma, tau=self.tau)
            traj['adv'] = advantages
            traj['ret'] = returns

        # Losses
        total_loss = 0.0

        # Leader (policy 0) gets on‑policy + off‑policy loss
        leader_traj = trajs[0]
        # On‑policy loss
        on_policy_loss = self._policy_loss(leader_traj, 0)
        total_loss += on_policy_loss

        # Off‑policy from followers
        off_policy_traj = self._merge_follower_data(trajs[1:])
        off_policy_loss = self._off_policy_loss(off_policy_traj, 0)
        total_loss += self.lambda_off * off_policy_loss

        # Followers only on‑policy
        for i in range(1, self.M):
            on_loss = self._policy_loss(trajs[i], i)
            total_loss += on_loss

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.shared_policy.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.conds.parameters(), 1.0)
        self.optimizer.step()

    def _policy_loss(self, traj, policy_id):
        # Compute ratio r_t = pi_theta / pi_old
        cond = self.conds[policy_id].expand_as(traj['obs'])
        mean, std = self.shared_policy.forward(traj['obs'], cond)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(traj['act']).sum(-1, keepdim=True)
        r = torch.exp(logp - traj['logp'])
        # Clipped surrogate
        clipped = torch.clamp(r, 1 - self.clip_eps, 1 + self.clip_eps)
        surrogate = torch.min(r, clipped) * traj['adv']
        loss = - surrogate.mean()
        # Entropy bonus
        entropy = dist.entropy().sum(-1).mean()
        loss -= self.entropy_coef * entropy
        return loss

    def _off_policy_loss(self, traj, policy_id):
        # Importance weight ratio r_i_j = pi_i / pi_j
        cond_i = self.conds[policy_id].expand_as(traj['obs'])
        mean_i, std_i = self.shared_policy.forward(traj['obs'], cond_i)
        dist_i = torch.distributions.Normal(mean_i, std_i)
        logp_i = dist_i.log_prob(traj['act']).sum(-1, keepdim=True)

        # For simplicity we approximate pi_j by the policy that generated the data
        # which is the same as policy j. We reconstruct pi_j using its cond.
        # Here we just reuse the same cond as in the original data, assuming
        # the data dict contains the cond index.
        # Since this is a toy demo we ignore the exact pi_j and set weight=1.
        # In a full implementation you would compute the ratio explicitly.
        r = 1.0

        clipped = torch.clamp(r, 1 - self.clip_eps, 1 + self.clip_eps)
        surrogate = torch.min(r, clipped) * traj['adv']
        loss = - surrogate.mean()
        return loss

    def _merge_follower_data(self, follower_trajs):
        """
        Concatenate data from all follower policies.
        """
        merged = {k: torch.cat([t[k] for t in follower_trajs], dim=0)
                  for k in follower_trajs[0].keys()}
        return merged

    def evaluate(self, num_episodes=10):
        """
        Evaluate each policy for a few episodes and return mean rewards.
        """
        mean_rewards = []
        for policy_id in range(self.M):
            total = 0.0
            for _ in range(num_episodes):
                obs, _ = self.envs[policy_id * self.envs_per_policy].reset()
                done = False
                ep_reward = 0.0
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32,
                                         device=self.device).unsqueeze(0)
                    cond = self.conds[policy_id].unsqueeze(0)
                    mean, std = self.shared_policy.forward(obs_t, cond)
                    action = mean.squeeze(0).cpu().numpy()
                    obs, reward, done, _, _ = self.envs[policy_id * self.envs_per_policy].step(action)
                    ep_reward += reward
                total += ep_reward
            mean_rewards.append(total / num_episodes)
        return mean_rewards