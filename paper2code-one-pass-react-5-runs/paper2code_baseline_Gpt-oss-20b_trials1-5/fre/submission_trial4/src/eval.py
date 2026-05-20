"""
Evaluate the trained FRE model on a new reward function.
"""

import os
import torch
import numpy as np
from tqdm import tqdm

from .dataset import OfflineDataset
from .fre import Encoder, Decoder

# ====================
#  Hyper‑parameters
# ====================
STATE_DIM = 2
LATENT_DIM = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 123

# ====================
#  Reward prior functions (same as in trainer)
# ====================
def random_goal_reward(samples, goal):
    return np.where(np.linalg.norm(samples - goal, axis=-1) < 0.1, 0.0, -1.0)

def random_linear_reward(samples, w, b=0.0):
    return np.dot(samples, w) + b

def random_mlp_reward(samples, rng):
    w1 = rng.normal(scale=1.0, size=(samples.shape[1], 32))
    b1 = rng.normal(scale=0.1, size=(32,))
    w2 = rng.normal(scale=1.0, size=(32, 1))
    b2 = rng.normal(scale=0.1, size=(1,))
    h = np.tanh(np.dot(samples, w1) + b1)
    out = np.dot(h, w2) + b2
    return out.squeeze(-1)

def sample_reward_function(rng):
    choice = rng.choice(['goal', 'linear', 'mlp'])
    if choice == 'goal':
        goal = rng.normal(size=STATE_DIM)
        return lambda s: random_goal_reward(s, goal), f"Goal at {goal}"
    if choice == 'linear':
        w = rng.normal(scale=1.0, size=STATE_DIM)
        return lambda s: random_linear_reward(s, w), f"Linear w={w}"
    return lambda s: random_mlp_reward(s, rng), "MLP"

# ====================
#  Evaluation
# ====================
def main():
    rng = np.random.default_rng(SEED)
    dataset = OfflineDataset(state_dim=STATE_DIM, action_dim=2,
                             num_trajectories=200, traj_len=50, seed=SEED)

    # Load model
    ckpt = torch.load('fre_model.pt', map_location=DEVICE)
    encoder = Encoder(state_dim=STATE_DIM, reward_dim=1, latent_dim=LATENT_DIM).to(DEVICE)
    decoder = Decoder(state_dim=STATE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    encoder.eval()
    decoder.eval()

    # Sample a new reward function
    reward_fn, desc = sample_reward_function(rng)

    # Context set (32 samples)
    context_states = dataset.states[:32]                       # (32, D)
    context_rewards = reward_fn(context_states)                # (32,)
    context_states_t = torch.tensor(context_states, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    context_rewards_t = torch.tensor(context_rewards, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        z, _, _ = encoder(context_states_t, context_rewards_t)   # (1, latent_dim)

    # Test set (10,000 random states)
    test_states = dataset.states[32:10332]                   # (10,000, D)
    test_rewards_true = reward_fn(test_states)                # (10,000,)
    test_states_t = torch.tensor(test_states, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    test_rewards_pred = decoder(test_states_t, z).squeeze(0).cpu().numpy()

    mse = np.mean((test_rewards_true - test_rewards_pred) ** 2)

    # Log results
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/results.txt', 'a') as f:
        f.write("\n=== Evaluation Results ===\n")
        f.write(f"Test Reward: {desc}\n")
        f.write(f"MSE on {len(test_states)} test states: {mse:.6f}\n")

    print("\n=== Evaluation Results ===")
    print(f"Test Reward: {desc}")
    print(f"MSE on {len(test_states)} test states: {mse:.6f}")

if __name__ == "__main__":
    main()