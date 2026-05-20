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

# ----------------------- Network definitions -----------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action, z):
        x = torch.cat([state, action, z], dim=-1)
        return self.net(x).squeeze(-1)

class VNetwork(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, z):
        x = torch.cat([state, z], dim=-1)
        return self.net(x).squeeze(-1)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # actions in [-1,1]
        )

    def forward(self, state, z):
        return self.net(torch.cat([state, z], dim=-1))

# ----------------------- Training routine -----------------------

def train_policy(fre_ckpt_path: str = "fre_checkpoint/fre_checkpoint.pt",
                 policy_ckpt_path: str = "policy_checkpoint",
                 dataset_name: str = "halfcheetah-medium-expert-v2",
                 total_steps: int = 850_000,
                 batch_size: int = 512,
                 K: int = 32,
                 lr: float = 1e-4,
                 gamma: float = 0.88,
                 hidden_dim: int = 512,
                 latent_dim: int = 32,
                 seed: int = 42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load FRE and freeze encoder
    ckpt = torch.load(fre_ckpt_path, map_location=device)
    cfg = ckpt["config"]
    state_dim = cfg["state_dim"]
    encoder = FREEncoder(state_dim,
                         hidden_dim=cfg["hidden_dim"],
                         num_layers=cfg["num_layers"],
                         num_heads=cfg["num_heads"],
                         latent_dim=latent_dim).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Load dataset
    dataset = load_dataset(dataset_name)
    obs = dataset["observations"]
    actions = dataset["actions"]
    next_obs = dataset["next_observations"]
    dones = dataset["terminals"]
    N = obs.shape[0]

    # Networks
    q1 = QNetwork(state_dim, actions.shape[-1], latent_dim, hidden_dim).to(device)
    q2 = QNetwork(state_dim, actions.shape[-1], latent_dim, hidden_dim).to(device)
    v = VNetwork(state_dim, latent_dim, hidden_dim).to(device)
    policy = PolicyNetwork(state_dim, actions.shape[-1], latent_dim, hidden_dim).to(device)

    # Target networks
    q1_target = QNetwork(state_dim, actions.shape[-1], latent_dim, hidden_dim).to(device)
    q2_target = QNetwork(state_dim, actions.shape[-1], latent_dim, hidden_dim).to(device)
    v_target = VNetwork(state_dim, latent_dim, hidden_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    v_target.load_state_dict(v.state_dict())

    # Optimizers
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    v_opt = optim.Adam(v.parameters(), lr=lr)
    p_opt = optim.Adam(policy.parameters(), lr=lr)

    # Replay buffer is simply the dataset indices
    indices = np.arange(N)

    for step in tqdm(range(total_steps), desc="Training Policy"):
        # Sample batch
        batch_idx = np.random.choice(indices, size=batch_size, replace=False)
        s = torch.from_numpy(obs[batch_idx]).float().to(device)
        a = torch.from_numpy(actions[batch_idx]).float().to(device)
        s_next = torch.from_numpy(next_obs[batch_idx]).float().to(device)
        d = torch.from_numpy(dones[batch_idx]).float().to(device)

        # Sample a random reward function for this batch
        reward_func = sample_reward_function(dataset, state_dim)
        r = torch.from_numpy(reward_func(s.cpu().numpy())).float().to(device)

        # Encode context states (drawn independently from dataset)
        ctx_idx = np.random.choice(indices, size=K, replace=False)
        ctx_states = torch.from_numpy(obs[ctx_idx]).float().to(device)
        ctx_rewards = torch.from_numpy(reward_func(ctx_states.cpu().numpy())).float().unsqueeze(-1).to(device)
        with torch.no_grad():
            z, _, _ = encoder(ctx_states.unsqueeze(0), ctx_rewards.unsqueeze(0))  # [1, latent]
        z = z.squeeze(0)  # [latent]

        # Predict next action from policy target
        with torch.no_grad():
            a_next = policy(s_next, z)
            q1_next = q1_target(s_next, a_next, z)
            q2_next = q2_target(s_next, a_next, z)
            q_next_min = torch.min(q1_next, q2_next)
            v_next = v_target(s_next, z)
            # IQL style target: r + gamma * v_next
            target_q = r + gamma * v_next

        # Critic loss
        q1_pred = q1(s, a, z)
        q2_pred = q2(s, a, z)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        q_loss = q1_loss + q2_loss

        # Value loss (clipped to advantage)
        v_pred = v(s, z)
        adv = q1_pred - v_pred
        loss_adv = torch.clamp(adv, min=-1.0, max=1.0)
        v_loss = F.mse_loss(v_pred, loss_adv.detach() + v_pred)

        # Policy loss (maximize Q)
        a_pred = policy(s, z)
        q1_pi = q1(s, a_pred, z)
        p_loss = -q1_pi.mean()  # maximize

        # Optimize
        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        p_opt.zero_grad()
        p_loss.backward()
        p_opt.step()

        # Soft updates
        for target, source in [(q1_target, q1), (q2_target, q2), (v_target, v)]:
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

        if step % 20_000 == 0:
            print(f"Step {step:5d} | q_loss={q_loss.item():.4f} | v_loss={v_loss.item():.4f} | p_loss={p_loss.item():.4f}")

    # Save policy checkpoint
    os.makedirs(policy_ckpt_path, exist_ok=True)
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "v_state_dict": v.state_dict(),
        "q1_state_dict": q1.state_dict(),
        "q2_state_dict": q2.state_dict(),
        "config": {
            "state_dim": state_dim,
            "action_dim": actions.shape[-1],
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "gamma": gamma,
            "total_steps": total_steps
        }
    }, os.path.join(policy_ckpt_path, "policy_checkpoint.pt"))
    print(f"Policy checkpoint written to {policy_ckpt_path}/policy_checkpoint.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fre_ckpt", default="fre_checkpoint/fre_checkpoint.pt",
                        help="path to FRE checkpoint")
    parser.add_argument("--policy_ckpt", default="policy_checkpoint",
                        help="directory to store policy checkpoint")
    parser.add_argument("--env", default="halfcheetah-medium-expert-v2",
                        help="d4rl environment name")
    parser.add_argument("--steps", type=int, default=850_000,
                        help="total training steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    args = parser.parse_args()
    train_policy(fre_ckpt_path=args.fre_ckpt,
                 policy_ckpt_path=args.policy_ckpt,
                 dataset_name=args.env,
                 total_steps=args.steps,
                 seed=args.seed)