"""
Training script for FRE (encoder + decoder).
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from .dataset import OfflineDataset
from .fre import Encoder, Decoder

# ====================
#  Hyper‑parameters
# ====================
STATE_DIM = 2
ACTION_DIM = 2
LATENT_DIM = 32
BATCH_SIZE = 256
ENC_K = 8          # number of context samples for encoder
DEC_K = 4          # number of decoding samples
LEARNING_RATE = 1e-3
NUM_ITERS = 2000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# ====================
#  Reward prior functions
# ====================
def random_goal_reward(samples, goal):
    # reward = 0 if state == goal else -1
    return np.where(np.linalg.norm(samples - goal, axis=-1) < 0.1, 0.0, -1.0)

def random_linear_reward(samples, w, b=0.0):
    return np.dot(samples, w) + b

def random_mlp_reward(samples, rng):
    # Two‑layer MLP with tanh
    w1 = rng.normal(scale=1.0, size=(samples.shape[1], 32))
    b1 = rng.normal(scale=0.1, size=(32,))
    w2 = rng.normal(scale=1.0, size=(32, 1))
    b2 = rng.normal(scale=0.1, size=(1,))
    h = np.tanh(np.dot(samples, w1) + b1)
    out = np.dot(h, w2) + b2
    return out.squeeze(-1)

def sample_reward_function(rng):
    """Return a callable reward function and a description."""
    choice = rng.choice(['goal', 'linear', 'mlp'])
    if choice == 'goal':
        goal = rng.normal(size=STATE_DIM)
        return lambda s: random_goal_reward(s, goal), f"Goal at {goal}"
    if choice == 'linear':
        w = rng.normal(scale=1.0, size=STATE_DIM)
        return lambda s: random_linear_reward(s, w), f"Linear w={w}"
    # mlp
    return lambda s: random_mlp_reward(s, rng), "MLP"

# ====================
#  Training loop
# ====================
def main():
    rng = np.random.default_rng(SEED)
    dataset = OfflineDataset(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                             num_trajectories=200, traj_len=50, seed=SEED)

    encoder = Encoder(state_dim=STATE_DIM, reward_dim=1,
                      latent_dim=LATENT_DIM).to(DEVICE)
    decoder = Decoder(state_dim=STATE_DIM, latent_dim=LATENT_DIM).to(DEVICE)

    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                     lr=LEARNING_RATE)

    loss_fn = torch.nn.MSELoss()

    # Log results
    os.makedirs('experiments', exist_ok=True)
    result_file = open('experiments/results.txt', 'w')

    for it in tqdm(range(NUM_ITERS), desc='Training FRE'):
        # Sample a batch of transitions
        batch = dataset.sample_batch(BATCH_SIZE, seed=rng.integers(1e9))
        states = torch.tensor(batch['states'], dtype=torch.float32, device=DEVICE)  # (B, D)

        # Sample a random reward function
        reward_fn, desc = sample_reward_function(rng)

        # Encode
        enc_states = states[:, :ENC_K]      # (B, K, D)
        enc_rewards = torch.tensor(reward_fn(enc_states.cpu().numpy()), dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        z, mu, logvar = encoder(enc_states, enc_rewards)

        # Decode
        dec_states = states[:, :DEC_K]      # (B, K, D)
        dec_rewards_true = torch.tensor(reward_fn(dec_states.cpu().numpy()), dtype=torch.float32, device=DEVICE)
        dec_rewards_pred = decoder(dec_states, z)

        # Loss
        recon_loss = loss_fn(dec_rewards_pred, dec_rewards_true)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.01 * kl_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 200 == 0:
            print(f"Iter {it}: loss={loss.item():.4f} recon={recon_loss.item():.4f} kl={kl_loss.item():.4f}")

    # Save the trained model
    torch.save({'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict()}, 'fre_model.pt')
    result_file.write("Training completed.\n")
    result_file.close()

if __name__ == "__main__":
    main()