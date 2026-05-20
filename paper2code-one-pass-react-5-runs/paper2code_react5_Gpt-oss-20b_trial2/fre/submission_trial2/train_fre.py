import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
from tqdm import tqdm
from fre import FREEncoder, FREDecoder
from dataset_loader import load_dataset
import torch.nn.functional as F

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------- Reward function factories -----------------------

class RewardFunction:
    def __call__(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class GoalReward(RewardFunction):
    def __init__(self, goal: np.ndarray, threshold: float = 0.2):
        self.goal = goal
        self.threshold = threshold

    def __call__(self, states: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(states - self.goal, axis=-1)
        return np.where(dists < self.threshold, 0.0, -1.0).astype(np.float32)

class LinearReward(RewardFunction):
    def __init__(self, w: np.ndarray, mask: np.ndarray = None):
        self.w = w
        self.mask = mask

    def __call__(self, states: np.ndarray) -> np.ndarray:
        return np.dot(states, self.w * (self.mask if self.mask is not None else 1.0)).astype(np.float32)

class MLPReward(RewardFunction):
    def __init__(self, net: nn.Module):
        self.net = net

    def __call__(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(states).float().to(next(self.net.parameters()).device)
            out = self.net(x).cpu().numpy().squeeze(-1)
        return out.astype(np.float32)

def random_goal_reward(dataset, threshold=0.2):
    idx = np.random.choice(len(dataset["observations"]))
    goal = dataset["observations"][idx]
    return GoalReward(goal, threshold)

def random_linear_reward(state_dim, sparsity=0.9):
    w = np.random.uniform(-1, 1, size=state_dim).astype(np.float32)
    mask = np.random.binomial(1, 1 - sparsity, size=state_dim).astype(np.float32)
    return LinearReward(w, mask)

def random_mlp_reward(state_dim, hidden_dim=32):
    net = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1)
    )
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return MLPReward(net)

def sample_reward_function(dataset, state_dim):
    rtype = random.choice(["goal", "linear", "mlp"])
    if rtype == "goal":
        return random_goal_reward(dataset)
    elif rtype == "linear":
        return random_linear_reward(state_dim)
    else:
        return random_mlp_reward(state_dim)

# ----------------------- Training routine -----------------------

def train_fre(dataset_name: str = "halfcheetah-medium-expert-v2",
              output_dir: str = "fre_checkpoint",
              total_steps: int = 150_000,
              batch_size: int = 32,
              K: int = 32,
              Kp: int = 8,
              beta: float = 0.01,
              lr: float = 1e-4,
              hidden_dim: int = 256,
              num_layers: int = 4,
              num_heads: int = 4,
              latent_dim: int = 32,
              seed: int = 42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(dataset_name)
    obs = dataset["observations"]
    state_dim = obs.shape[1]

    # Models
    encoder = FREEncoder(state_dim,
                         hidden_dim=hidden_dim,
                         num_layers=num_layers,
                         num_heads=num_heads,
                         latent_dim=latent_dim).to(device)
    decoder = FREDecoder(state_dim, latent_dim, hidden_dim).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    for step in tqdm(range(total_steps), desc="Training FRE"):
        reward_func = sample_reward_function(dataset, state_dim)

        # Encoder samples
        idx_enc = np.random.choice(len(obs), size=K, replace=False)
        states_enc = obs[idx_enc]
        rewards_enc = reward_func(states_enc)[:, None]  # [K,1]

        # Decoder samples
        idx_dec = np.random.choice(len(obs), size=Kp, replace=False)
        states_dec = obs[idx_dec]
        rewards_dec = reward_func(states_dec)[:, None]

        # Tensors
        states_enc_t = torch.from_numpy(states_enc).float().unsqueeze(0).to(device)
        rewards_enc_t = torch.from_numpy(rewards_enc).float().unsqueeze(0).to(device)
        states_dec_t = torch.from_numpy(states_dec).float().unsqueeze(0).to(device)
        rewards_dec_t = torch.from_numpy(rewards_dec).float().unsqueeze(0).to(device)

        # Forward
        z, mean, logvar = encoder(states_enc_t, rewards_enc_t)
        preds = decoder(states_dec_t, z)

        # Loss
        mse = F.mse_loss(preds, rewards_dec_t)
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = mse + beta * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10_000 == 0:
            print(f"Step {step:5d} | loss={loss.item():.4f} | mse={mse.item():.4f} | kl={kl.item():.4f}")

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "config": {
            "state_dim": state_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "latent_dim": latent_dim,
            "K": K,
            "Kp": Kp,
            "beta": beta,
            "lr": lr,
            "total_steps": total_steps
        }
    }, os.path.join(output_dir, "fre_checkpoint.pt"))
    print(f"FRE checkpoint written to {output_dir}/fre_checkpoint.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-expert-v2",
                        help="d4rl environment name")
    parser.add_argument("--output", default="fre_checkpoint",
                        help="directory to store checkpoint")
    parser.add_argument("--steps", type=int, default=150_000,
                        help="total training steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    args = parser.parse_args()
    train_fre(dataset_name=args.env,
              output_dir=args.output,
              total_steps=args.steps,
              seed=args.seed)