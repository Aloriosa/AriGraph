import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from src.dataset_loader import load_offline_dataset, get_batches
from src.reward_funcs import sample_random_reward
from src.fre import FreEncoder, FreDecoder
from src.policy import PolicyNet

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

STATE_DIM = 4      # CartPole state dimension
ACTION_DIM = 1     # CartPole discrete actions (treated as continuous in [-1,1])
LATENT_DIM = 32
BATCH_SIZE = 256
K_ENCODER = 32     # number of state‑reward pairs for encoding
K_DECODER = 8      # number of decoder states per reward
ENC_EPOCHS = 10
POL_EPOCHS = 20
LEARNING_RATE = 1e-3
BETA = 0.01        # KL weight for VAE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAMMA = 0.99

# ------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------
DATA_PATH = os.path.join('data', 'cartpole_offline.pkl')
dataset = load_offline_dataset(DATA_PATH)

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
encoder = FreEncoder(state_dim=STATE_DIM, latent_dim=LATENT_DIM,
                     n_layers=4, n_heads=4, embed_dim=128).to(DEVICE)
decoder = FreDecoder(state_dim=STATE_DIM, latent_dim=LATENT_DIM,
                     hidden_dim=256).to(DEVICE)
policy_net = PolicyNet(state_dim=STATE_DIM, latent_dim=LATENT_DIM,
                       action_dim=ACTION_DIM, hidden_dim=256).to(DEVICE)
q_net = nn.Sequential(
    nn.Linear(STATE_DIM + ACTION_DIM + LATENT_DIM, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(DEVICE)

# Optimizers
opt_enc_dec = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                         lr=LEARNING_RATE)
opt_policy = optim.Adam(list(policy_net.parameters()) + list(q_net.parameters()),
                        lr=LEARNING_RATE)

# ------------------------------------------------------------------
# Helper to sample K states from a batch
# ------------------------------------------------------------------
def sample_k_states(obs_batch, next_obs_batch, k):
    """Randomly sample k states from a batch (using the current states)."""
    idx = np.random.choice(len(obs_batch), k, replace=False)
    return obs_batch[idx], next_obs_batch[idx]

# ------------------------------------------------------------------
# Encoder‑Decoder training
# ------------------------------------------------------------------
print("Training FRE encoder/decoder...")
for epoch in range(ENC_EPOCHS):
    pbar = tqdm(get_batches(dataset, BATCH_SIZE), total=len(dataset)//BATCH_SIZE)
    for obs, _, next_obs, _ in pbar:
        obs = torch.from_numpy(obs).to(DEVICE)          # (B, state_dim)
        next_obs = torch.from_numpy(next_obs).to(DEVICE)

        # Sample K encoder states and compute rewards for a random reward function
        rng = np.random.default_rng()
        reward_fn, _ = sample_random_reward(STATE_DIM, rng)
        enc_states_np, _ = sample_k_states(obs.cpu().numpy(), next_obs.cpu().numpy(), K_ENCODER)
        enc_states = torch.from_numpy(enc_states_np).to(DEVICE)
        enc_rewards_np = np.array([reward_fn(s) for s in enc_states_np], dtype=np.float32)
        enc_rewards = torch.from_numpy(enc_rewards_np).unsqueeze(-1).to(DEVICE)

        # Encode to latent z
        mean, std = encoder(enc_states.unsqueeze(0), enc_rewards.unsqueeze(0))
        z = mean  # deterministic mean for simplicity

        # Sample K decoder states
        dec_states_np, _ = sample_k_states(obs.cpu().numpy(), next_obs.cpu().numpy(), K_DECODER)
        dec_states = torch.from_numpy(dec_states_np).to(DEVICE)
        true_rewards_np = np.array([reward_fn(s) for s in dec_states_np], dtype=np.float32)
        true_rewards = torch.from_numpy(true_rewards_np).unsqueeze(-1).to(DEVICE)

        # Decode
        pred_rewards = decoder(dec_states, z.expand(dec_states.shape[0], -1))

        # Loss: MSE + KL
        recon_loss = F.mse_loss(pred_rewards, true_rewards)
        kl = 0.5 * torch.mean(
            torch.sum(std**2 + mean**2 - 1.0 - torch.log(std**2), dim=-1)
        )
        loss = recon_loss + BETA * kl

        opt_enc_dec.zero_grad()
        loss.backward()
        opt_enc_dec.step()

        pbar.set_postfix({'enc-dec loss': loss.item()})

# Freeze encoder after training
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

# ------------------------------------------------------------------
# Policy training (SAC‑style offline RL conditioned on z)
# ------------------------------------------------------------------
print("Training policy network...")
for epoch in range(POL_EPOCHS):
    pbar = tqdm(get_batches(dataset, BATCH_SIZE), total=len(dataset)//BATCH_SIZE)
    for obs, actions, next_obs, _ in pbar:
        obs = torch.from_numpy(obs).to(DEVICE)          # (B, state_dim)
        actions = torch.from_numpy(actions).unsqueeze(-1).float().to(DEVICE)  # (B, 1)
        next_obs = torch.from_numpy(next_obs).to(DEVICE)

        # Sample random reward function for this batch
        rng = np.random.default_rng()
        reward_fn, _ = sample_random_reward(STATE_DIM, rng)

        # Compute reward for each transition (current state)
        obs_np = obs.cpu().numpy()
        rewards_np = np.array([reward_fn(s) for s in obs_np], dtype=np.float32)
        rewards = torch.from_numpy(rewards_np).unsqueeze(-1).to(DEVICE)

        # Encode states to latent z (use a subset of states from the same batch)
        enc_states_np, _ = sample_k_states(obs.cpu().numpy(), next_obs.cpu().numpy(), K_ENCODER)
        enc_states = torch.from_numpy(enc_states_np).to(DEVICE)
        enc_rewards_np = np.array([reward_fn(s) for s in enc_states_np], dtype=np.float32)
        enc_rewards = torch.from_numpy(enc_rewards_np).unsqueeze(-1).to(DEVICE)
        mean, std = encoder(enc_states.unsqueeze(0), enc_rewards.unsqueeze(0))
        z = mean.expand(obs.shape[0], -1)  # (B, latent_dim)

        # Q‑value target: r + γ * max_a' Q(s', a', z)
        with torch.no_grad():
            # Evaluate Q for both discrete actions 0 and 1
            a0 = torch.zeros((obs.shape[0], 1), device=DEVICE)
            a1 = torch.ones((obs.shape[0], 1), device=DEVICE)
            q_next_a0 = q_net(torch.cat([next_obs, a0, z], dim=-1))
            q_next_a1 = q_net(torch.cat([next_obs, a1, z], dim=-1))
            q_next_max = torch.max(q_next_a0, q_next_a1)
            target_q = rewards + GAMMA * q_next_max

        # Current Q
        cur_q = q_net(torch.cat([obs, actions, z], dim=-1))

        # Q loss
        loss_q = F.mse_loss(cur_q, target_q)

        # Policy loss: maximize Q(s, policy(s, z), z)
        act_pred = policy_net(obs, z)  # continuous in [-1,1]
        cur_q_policy = q_net(torch.cat([obs, act_pred, z], dim=-1))
        loss_policy = -cur_q_policy.mean()

        # Total loss
        loss = loss_q + loss_policy

        opt_policy.zero_grad()
        loss.backward()
        opt_policy.step()

        pbar.set_postfix({'policy loss': loss.item()})

# ------------------------------------------------------------------
# Save checkpoints
# ------------------------------------------------------------------
os.makedirs('checkpoints', exist_ok=True)
torch.save(encoder.state_dict(), 'checkpoints/encoder.pt')
torch.save(decoder.state_dict(), 'checkpoints/decoder.pt')
torch.save(policy_net.state_dict(), 'checkpoints/policy.pt')
torch.save(q_net.state_dict(), 'checkpoints/q.pt')

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
print("Evaluating on 10 random downstream tasks...")
num_tasks = 10
task_returns = []

for t in range(num_tasks):
    rng = np.random.default_rng()
    reward_fn, _ = sample_random_reward(STATE_DIM, rng)

    # Sample K encoder states from the dataset to encode this task
    idx = np.random.choice(len(dataset), K_ENCODER, replace=False)
    enc_states = np.vstack([dataset[i][0] for i in idx])  # (K, state_dim)
    enc_rewards_np = np.array([reward_fn(s) for s in enc_states], dtype=np.float32).reshape(-1, 1)
    enc_states_t = torch.from_numpy(enc_states).float().to(DEVICE)
    enc_rewards_t = torch.from_numpy(enc_rewards_np).float().to(DEVICE)

    z = encoder(enc_states_t.unsqueeze(0), enc_rewards_t.unsqueeze(0))[0]  # (1, latent_dim)
    z = z.expand(1, -1)

    # Rollout 5 episodes
    total_ret = 0.0
    for ep in range(5):
        env = __import__('gym').make('CartPole-v1')
        obs = env.reset(seed=SEED + t*10 + ep)
        ep_ret = 0.0
        done = False
        step = 0
        while not done and step < 200:
            # Current state reward
            ep_ret += reward_fn(obs)

            # Select action
            obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(DEVICE)
            act = policy_net(obs_t, z).detach().cpu().numpy()[0,0]
            act_int = int(act > 0.0)  # threshold to discrete action
            next_obs, _, done, _ = env.step(act_int)
            obs = next_obs
            step += 1
        env.close()
        total_ret += ep_ret
    avg_ret = total_ret / 5.0
    task_returns.append(avg_ret)

# Save results
import csv
with open('metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['task_id', 'average_return'])
    for i, r in enumerate(task_returns):
        writer.writerow([i, r])
    writer.writerow(['overall_mean', np.mean(task_returns)])

print(f"Average return over {num_tasks} tasks: {np.mean(task_returns):.3f}")