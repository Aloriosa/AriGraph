# -*- coding: utf-8 -*-
"""
Core implementation of Functional Reward Encoding (FRE).
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#  Hyper‑parameters (close to those in the paper)
# --------------------------------------------------------------------------- #

BATCH_SIZE = 512
ENCODER_STEPS = 10000
DECODER_STEPS = 8
CONTEXT_SAMPLES = 32
DECODER_SAMPLES = 8
LATENT_DIM = 128
REWARD_EMBED_DIM = 128
REWARD_EMBED_BINS = 32
ENCODER_LAYERS = 4
ENCODER_HEADS = 4
ENCODER_HIDDEN = 256
DECODER_HIDDEN = 512
KL_BETA = 0.01
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------------------------------- #
#  Reward Prior
# --------------------------------------------------------------------------- #

class RewardPrior:
    """
    Sample random reward functions from three families:
    1. Goal‑reaching (singleton)
    2. Linear
    3. 2‑layer MLP
    """

    def __init__(self, dataset_obs: np.ndarray):
        """
        Args:
            dataset_obs: (N, state_dim) array of observations from the offline dataset.
        """
        self.dataset_obs = dataset_obs
        self.state_dim = dataset_obs.shape[1]

    def sample(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Randomly choose a reward family and return a callable that maps states -> reward.
        """
        choice = random.choice(["goal", "linear", "mlp"])
        if choice == "goal":
            return self._sample_goal_reward()
        elif choice == "linear":
            return self._sample_linear_reward()
        else:
            return self._sample_mlp_reward()

    def _sample_goal_reward(self) -> Callable[[np.ndarray], np.ndarray]:
        # Sample a random goal state from the dataset
        goal = self.dataset_obs[np.random.randint(0, len(self.dataset_obs))]
        # Use only the first two state dims (position) for distance
        goal_pos = goal[:2]

        def reward(s: np.ndarray) -> np.ndarray:
            # s can be (batch, state_dim)
            pos = s[:, :2]
            dist = np.linalg.norm(pos - goal_pos, axis=1)
            return np.where(dist < 2.0, 0.0, -1.0)

        return reward

    def _sample_linear_reward(self) -> Callable[[np.ndarray], np.ndarray]:
        # Random vector with sparsity
        mask = np.random.binomial(1, 0.1, size=self.state_dim).astype(np.float32)
        vec = np.random.uniform(-1.0, 1.0, size=self.state_dim).astype(np.float32)
        vec = vec * mask
        vec = vec / (np.linalg.norm(vec) + 1e-8)  # normalise

        def reward(s: np.ndarray) -> np.ndarray:
            return np.dot(s, vec)

        return reward

    def _sample_mlp_reward(self) -> Callable[[np.ndarray], np.ndarray]:
        # Two‑layer MLP with tanh
        hidden_dim = 32
        w1 = np.random.randn(self.state_dim, hidden_dim).astype(np.float32) * 0.1
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        w2 = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.1
        b2 = np.zeros(1, dtype=np.float32)

        def reward(s: np.ndarray) -> np.ndarray:
            h = np.tanh(np.dot(s, w1) + b1)
            out = np.dot(h, w2) + b2
            return np.clip(out.squeeze(-1), -1.0, 1.0)

        return reward


# --------------------------------------------------------------------------- #
#  Encoder / Decoder
# --------------------------------------------------------------------------- #

class FREEncoder(nn.Module):
    """
    Permutation‑invariant transformer encoder that maps a set of
    (state, reward) pairs to a latent Gaussian distribution.
    """

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim

        # Project state to hidden dim
        self.state_proj = nn.Linear(state_dim, ENCODER_HIDDEN, bias=False)

        # Reward embedding (discretised)
        self.reward_embed = nn.Embedding(REWARD_EMBED_BINS, REWARD_EMBED_DIM)

        # Positional encoding is not used (set to zero)
        self.input_proj = nn.Linear(ENCODER_HIDDEN + REWARD_EMBED_DIM,
                                    ENCODER_HIDDEN)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ENCODER_HIDDEN,
            nhead=ENCODER_HEADS,
            dim_feedforward=ENCODER_HIDDEN * 4,
            dropout=0.0,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=ENCODER_LAYERS
        )

        # Output mean and logvar
        self.mean_head = nn.Linear(ENCODER_HIDDEN, LATENT_DIM)
        self.logvar_head = nn.Linear(ENCODER_HIDDEN, LATENT_DIM)

    def forward(self, states: torch.Tensor, rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (batch, K, state_dim)
            rewards: (batch, K)  values in [-1,1]
        Returns:
            mean, logvar  (batch, LATENT_DIM)
        """
        # Map rewards to discrete bins
        eps = 1e-6
        r_norm = (rewards + 1.0) / 2.0  # [0,1]
        bin_idx = torch.clamp((r_norm * REWARD_EMBED_BINS).long(), 0, REWARD_EMBED_BINS - 1)
        r_emb = self.reward_embed(bin_idx)  # (batch, K, REWARD_EMBED_DIM)

        # Project states
        s_emb = self.state_proj(states)  # (batch, K, ENCODER_HIDDEN)

        # Concatenate
        x = torch.cat([s_emb, r_emb], dim=-1)  # (batch, K, ENCODER_HIDDEN)

        # Project to model dim
        x = self.input_proj(x)

        # Transformer expects (K, batch, dim)
        x = x.permute(1, 0, 2)

        x = self.transformer(x)  # (K, batch, dim)

        # Aggregate (mean over set)
        x = x.mean(dim=0)  # (batch, dim)

        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        return mean, logvar


class FREDecoder(nn.Module):
    """
    MLP that predicts reward given (state, latent).
    """

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + LATENT_DIM, DECODER_HIDDEN),
            nn.GELU(),
            nn.Linear(DECODER_HIDDEN, DECODER_HIDDEN),
            nn.GELU(),
            nn.Linear(DECODER_HIDDEN, 1),
        )

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch, K', state_dim)
            z: (batch, LATENT_DIM)
        Returns:
            reward predictions: (batch, K')
        """
        batch, Kp, _ = states.shape
        z_exp = z.unsqueeze(1).expand(-1, Kp, -1).reshape(batch * Kp, LATENT_DIM)
        inp = torch.cat([states.reshape(batch * Kp, -1), z_exp], dim=-1)
        out = self.net(inp).reshape(batch, Kp)
        return out.squeeze(-1)


# --------------------------------------------------------------------------- #
#  Utility functions
# --------------------------------------------------------------------------- #

def discretise_rewards(rewards: torch.Tensor) -> torch.Tensor:
    """
    Map continuous rewards in [-1,1] to discrete bins in [0, REWARD_EMBED_BINS-1].
    """
    r_norm = (rewards + 1.0) / 2.0
    bin_idx = torch.clamp((r_norm * REWARD_EMBED_BINS).long(), 0, REWARD_EMBED_BINS - 1)
    return bin_idx


def reparameterise(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between N(mean, var) and N(0, I)
    """
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1).mean()


# --------------------------------------------------------------------------- #
#  Training loop for encoder
# --------------------------------------------------------------------------- #

def train_encoder(
    env_name: str,
    encoder_steps: int,
    batch_size: int = BATCH_SIZE,
    device: torch.device = DEVICE,
):
    """
    Train the FRE encoder on the specified offline dataset.
    """
    print(f"Loading dataset for {env_name} ...")
    env = gym.make(env_name)
    dataset = env.get_dataset()
    obs = dataset["observations"]  # (N, state_dim)
    act = dataset["actions"]       # (N, act_dim)
    rew = dataset["rewards"]       # (N,)
    next_obs = dataset["next_observations"]  # (N, state_dim)
    done = dataset["terminals"]    # (N,)

    state_dim = obs.shape[1]
    encoder = FREEncoder(state_dim).to(device)
    decoder = FREDecoder(state_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=LR
    )

    reward_prior = RewardPrior(obs)

    dataset_size = obs.shape[0]
    indices = np.arange(dataset_size)

    pbar = tqdm(range(encoder_steps), desc="Encoder training")
    for step in pbar:
        # Sample reward function
        reward_fn = reward_prior.sample()

        # Sample K context states
        ctx_idx = np.random.choice(indices, size=CONTEXT_SAMPLES, replace=False)
        ctx_states = torch.tensor(obs[ctx_idx], dtype=torch.float32, device=device)  # (K, state_dim)
        ctx_rewards_np = reward_fn(ctx_states.cpu().numpy())  # (K,)
        ctx_rewards = torch.tensor(ctx_rewards_np, dtype=torch.float32, device=device)

        # Sample K' target states
        tgt_idx = np.random.choice(indices, size=DECODER_SAMPLES, replace=False)
        tgt_states = torch.tensor(obs[tgt_idx], dtype=torch.float32, device=device)  # (K', state_dim)
        tgt_rewards_np = reward_fn(tgt_states.cpu().numpy())
        tgt_rewards = torch.tensor(tgt_rewards_np, dtype=torch.float32, device=device)

        # Encode
        mean, logvar = encoder(ctx_states.unsqueeze(0), ctx_rewards.unsqueeze(0))  # (1, LATENT_DIM)
        z = reparameterise(mean, logvar).squeeze(0)  # (LATENT_DIM,)

        # Decode
        pred = decoder(tgt_states.unsqueeze(0), z.unsqueeze(0)).squeeze(0)  # (K',)

        # Loss
        mse = F.mse_loss(pred, tgt_rewards)
        kl = kl_divergence(mean, logvar)
        loss = mse + KL_BETA * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{mse.item():.4f}",
                kl=f"{kl.item():.4f}",
            )

    print("Encoder training finished.")
    return encoder, decoder, env


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

def evaluate_encoder(encoder: FREEncoder, env, device: torch.device = DEVICE):
    """
    Evaluate the trained encoder on three downstream reward functions.
    """
    print("\n=== Zero‑Shot Evaluation (MSE) ===")
    obs = env.get_dataset()["observations"]
    state_dim = obs.shape[1]
    # Create reward priors for each downstream task
    reward_fns = {
        "Goal‑reaching": RewardPrior(obs).sample_goal_reward(),
        "Linear reward": RewardPrior(obs).sample_linear_reward(),
        "MLP reward": RewardPrior(obs).sample_mlp_reward(),
    }

    results = {}
    for name, reward_fn in reward_fns.items():
        # Sample 64 states for evaluation
        eval_idx = np.random.choice(len(obs), size=64, replace=False)
        eval_states = torch.tensor(obs[eval_idx], dtype=torch.float32, device=device)

        # Compute true rewards
        true_rewards = torch.tensor(
            reward_fn(eval_states.cpu().numpy()), dtype=torch.float32, device=device
        )

        # Encode using K context states (random)
        ctx_idx = np.random.choice(len(obs), size=CONTEXT_SAMPLES, replace=False)
        ctx_states = torch.tensor(obs[ctx_idx], dtype=torch.float32, device=device)
        ctx_rewards_np = reward_fn(ctx_states.cpu().numpy())
        ctx_rewards = torch.tensor(ctx_rewards_np, dtype=torch.float32, device=device)

        mean, logvar = encoder(ctx_states.unsqueeze(0), ctx_rewards.unsqueeze(0))
        z = reparameterise(mean, logvar).squeeze(0)

        # Predict rewards for eval_states
        pred_rewards = []
        batch_size = 16
        for i in range(0, len(eval_states), batch_size):
            batch_states = eval_states[i : i + batch_size].unsqueeze(0)
            pred = encoder.decoder(batch_states, z.unsqueeze(0)).squeeze(0)
            pred_rewards.append(pred)
        pred_rewards = torch.cat(pred_rewards, dim=0)

        mse = F.mse_loss(pred_rewards, true_rewards).item()
        results[name] = mse
        print(f"{name:15s} : {mse:.4f}")

    return results


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="antmaze-large-diverse-v2")
    parser.add_argument("--encoder-steps", type=int, default=ENCODER_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    encoder, decoder, env = train_encoder(
        args.env, args.encoder_steps, args.batch_size, DEVICE
    )

    # Save encoder
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(encoder.state_dict(), f"{args.output_dir}/encoder.pt")
    torch.save(decoder.state_dict(), f"{args.output_dir}/decoder.pt")

    # Evaluation
    eval_results = evaluate_encoder(encoder, env, DEVICE)

    # Write results
    with open(f"{args.output_dir}/results.txt", "w") as f:
        f.write("=== Zero‑Shot Evaluation (MSE) ===\n")
        for name, mse in eval_results.items():
            f.write(f"{name:15s} : {mse:.4f}\n")