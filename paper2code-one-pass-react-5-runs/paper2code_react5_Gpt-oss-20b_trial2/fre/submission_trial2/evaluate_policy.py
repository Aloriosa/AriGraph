import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import gymnasium as gym
import d4rl
from fre import FREEncoder, FREDecoder
from dataset_loader import load_dataset
from train_policy import PolicyNetwork, VNetwork

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Reward functions same as earlier
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

def evaluate_policy(policy_ckpt_path: str = "policy_checkpoint/policy_checkpoint.pt",
                    fre_ckpt_path: str = "fre_checkpoint/fre_checkpoint.pt",
                    env_name: str = "halfcheetah-medium-expert-v2",
                    eval_episodes: int = 5,
                    seed: int = 42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load FRE
    fre_ckpt = torch.load(fre_ckpt_path, map_location=device)
    cfg = fre_ckpt["config"]
    state_dim = cfg["state_dim"]
    encoder = FREEncoder(state_dim,
                         hidden_dim=cfg["hidden_dim"],
                         num_layers=cfg["num_layers"],
                         num_heads=cfg["num_heads"],
                         latent_dim=cfg["latent_dim"]).to(device)
    encoder.load_state_dict(fre_ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Load policy
    ckpt = torch.load(policy_ckpt_path, map_location=device)
    policy_cfg = ckpt["config"]
    policy = PolicyNetwork(state_dim,
                           policy_cfg["action_dim"],
                           policy_cfg["latent_dim"],
                           hidden_dim=policy_cfg["hidden_dim"]).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # Environment
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Dataset for context states
    dataset = load_dataset(env_name)

    returns = []

    for ep in range(eval_episodes):
        # Sample a random reward function
        reward_func = sample_reward_function(dataset, state_dim)

        # Encode context states
        K = 32
        ctx_idx = np.random.choice(len(dataset["observations"]), size=K, replace=False)
        ctx_states = torch.from_numpy(dataset["observations"][ctx_idx]).float().to(device)
        ctx_rewards = torch.from_numpy(reward_func(ctx_states.cpu().numpy())).float().unsqueeze(-1).to(device)
        with torch.no_grad():
            z, _, _ = encoder(ctx_states.unsqueeze(0), ctx_rewards.unsqueeze(0))
        z = z.squeeze(0)

        # Run episode
        obs, _ = env.reset(seed=seed)
        ep_return = 0.0
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(obs_t, z).cpu().numpy().squeeze(0)
            action = np.clip(action, -max_action, max_action)
            next_obs, _, terminated, truncated, _ = env.step(action)
            reward = reward_func(np.array([next_obs]))[0]
            ep_return += reward
            obs = next_obs
            done = terminated or truncated
        returns.append(ep_return)
        print(f"Episode {ep+1}/{eval_episodes} | Return: {ep_return:.2f}")

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    out = {
        "reward_type": type(reward_func).__name__,
        "returns": returns,
        "average_return": float(avg_return),
        "std_return": float(std_return)
    }
    # Save results
    out_path = "policy_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Evaluation results written to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpt", default="policy_checkpoint/policy_checkpoint.pt",
                        help="path to policy checkpoint")
    parser.add_argument("--fre_ckpt", default="fre_checkpoint/fre_checkpoint.pt",
                        help="path to FRE checkpoint")
    parser.add_argument("--env", default="halfcheetah-medium-expert-v2",
                        help="environment name")
    parser.add_argument("--episodes", type=int, default=5,
                        help="number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    args = parser.parse_args()
    evaluate_policy(policy_ckpt_path=args.policy_ckpt,
                    fre_ckpt_path=args.fre_ckpt,
                    env_name=args.env,
                    eval_episodes=args.episodes,
                    seed=args.seed)