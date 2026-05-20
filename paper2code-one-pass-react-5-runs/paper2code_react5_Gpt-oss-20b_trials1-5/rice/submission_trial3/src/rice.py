import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from .mask_network import MaskNet
from .rnd import RNDTarget, RNDPredictor
from .env_utils import set_env_state

class RICE:
    """
    A minimal implementation of the RICE framework that demonstrates
    the core ideas:
        * Pre‑train a PPO policy.
        * Train a mask network that predicts the importance of each step.
        * Use Random Network Distillation to provide an exploration bonus.
        * Refine the policy by resetting the environment to a mixture of
          default initial states and critical states identified by the mask.
    The implementation is intentionally lightweight and focuses on
    reproducibility rather than matching the full experimental protocol.
    """

    def __init__(
        self,
        env_name: str,
        pretrained_path: str | None = None,
        lr: float = 3e-4,
        batch_size: int = 64,
        gamma: float = 0.99,
        p: float = 0.25,
        lam: float = 0.01,
        alpha: float = 0.0001,
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.env_name = env_name
        self.device = torch.device(device)
        self.p = p
        self.lam = lam
        self.alpha = alpha

        # Vectorised environment for Stable‑Baselines3
        self.env = DummyVecEnv([lambda: gym.make(env_name)])
        self.env.seed(seed)  # reproducibility

        # Policy – PPO with a small MLP
        self.policy = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=lr,
            batch_size=batch_size,
            gamma=gamma,
            verbose=0,
            device=self.device,
        )
        if pretrained_path and os.path.exists(pretrained_path):
            self.policy.load(pretrained_path)

        # Dimensions
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.state_dim = obs_space.shape[0]
        self.action_dim = (
            act_space.n if isinstance(act_space, gym.spaces.Discrete) else act_space.shape[0]
        )

        # Mask network
        self.mask_net = MaskNet(self.state_dim).to(self.device)
        self.mask_opt = optim.Adam(self.mask_net.parameters(), lr=lr)

        # RND
        self.rnd_target = RNDTarget(self.state_dim).to(self.device)
        for p_ in self.rnd_target.parameters():
            p_.requires_grad = False
        self.rnd_pred = RNDPredictor(self.state_dim).to(self.device)
        self.rnd_opt = optim.Adam(self.rnd_pred.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Utility: collect a single trajectory
    # ------------------------------------------------------------------
    def _sample_trajectory(self, max_steps: int = 200):
        obs = self.env.reset()
        traj = []
        for _ in tqdm(range(max_steps), leave=False):
            action, _ = self.policy.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            traj.append((obs[0], action[0], reward[0], next_obs[0]))
            obs = next_obs
            if terminated or truncated:
                break
        return traj

    # ------------------------------------------------------------------
    # Train the mask network (simple surrogate objective)
    # ------------------------------------------------------------------
    def train_mask(self, epochs: int = 5, samples_per_epoch: int = 500):
        for epoch in range(epochs):
            traj = self._sample_trajectory()
            states = torch.tensor([t[0] for t in traj], dtype=torch.float32).to(self.device)
            rewards = torch.tensor([t[2] for t in traj], dtype=torch.float32).to(self.device)

            # Mask probabilities
            mask_probs = self.mask_net(states).squeeze()
            # Sample binary mask
            mask_sample = torch.bernoulli(mask_probs).detach()

            # RND bonus
            next_states = torch.tensor([t[3] for t in traj], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                target_feat = self.rnd_target(next_states)
            pred_feat = self.rnd_pred(next_states)
            rnd_bonus = ((pred_feat - target_feat) ** 2).mean(dim=1)

            # Objective: maximize (reward + alpha * mask) - lam * rnd_bonus
            total_reward = rewards + self.alpha * mask_sample
            loss = -(total_reward.mean() - self.lam * rnd_bonus.mean())

            self.mask_opt.zero_grad()
            loss.backward()
            self.mask_opt.step()

    # ------------------------------------------------------------------
    # Pick a critical state from a trajectory
    # ------------------------------------------------------------------
    def _pick_critical_state(self, traj):
        states = torch.tensor([t[0] for t in traj], dtype=torch.float32).to(self.device)
        mask_probs = self.mask_net(states).squeeze()
        idx = mask_probs.argmax().item()
        return traj[idx][0]

    # ------------------------------------------------------------------
    # Main refinement loop
    # ------------------------------------------------------------------
    def refine(self, iter_steps: int = 200, num_iters: int = 200):
        for it in tqdm(range(num_iters), desc="RICE refinement"):
            # Decide whether to reset to a critical state
            if np.random.rand() < self.p:
                traj = self._sample_trajectory()
                crit_state = self._pick_critical_state(traj)
                try:
                    set_env_state(self.env.envs[0], crit_state)
                    self.env.reset()
                except Exception:
                    # Fallback to normal reset
                    self.env.reset()
            else:
                self.env.reset()

            # Collect a batch of experiences
            obs = self.env.reset()
            for _ in range(iter_steps):
                action, _ = self.policy.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # RND exploration bonus
                state_next = torch.tensor(next_obs[0], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    target_feat = self.rnd_target(state_next)
                pred_feat = self.rnd_pred(state_next)
                rnd_bonus = ((pred_feat - target_feat) ** 2).mean()
                reward = reward + self.lam * rnd_bonus.item()

                # Store transition in the replay buffer
                self.policy.replay_buffer.add(
                    obs[0], action[0], reward[0], next_obs[0], terminated[0]
                )
                obs = next_obs
                if terminated or truncated:
                    break

            # One PPO update step
            self.policy.train()

        # Save the refined policy
        self.policy.save("rice_refined.zip")
        return self.policy

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------
    def evaluate(self, episodes: int = 10):
        total = 0.0
        for _ in range(episodes):
            obs = self.env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                action, _ = self.policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_ret += reward[0]
                done = terminated or truncated
            total += ep_ret
        return total / episodes