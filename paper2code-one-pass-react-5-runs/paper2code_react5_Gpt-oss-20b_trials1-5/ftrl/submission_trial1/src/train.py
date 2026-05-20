# src/train.py
"""
Training script for the toy environment.
"""

import argparse
import os
import random
import json
from collections import deque, defaultdict
import numpy as np
import torch
import torch.optim as optim

from env import AppleRetrievalEnv
from policy import ActorCritic
from utils import compute_fisher


# ==============================
# Helper functions
# ==============================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def discount_cumsum(x, gamma):
    """Compute discounted cumulative sums of vectors."""
    return np.array([np.sum(x[: t + 1] * (gamma ** np.arange(t + 1))[::-1])
                     for t in range(len(x))])


def kl_divergence(p, q):
    """KL(p || q) for categorical distributions."""
    eps = 1e-8
    p = p.clamp(min=eps, max=1 - eps)
    q = q.clamp(min=eps, max=1 - eps)
    return (p * (p.log() - q.log())).sum(-1)


# ==============================
# PPO training helpers
# ==============================
def rollout(policy, env, device, max_steps, gamma=0.99, lam=0.95):
    """
    Collect a single trajectory using the current policy.
    Returns dict with obs, actions, rewards, dones, values, log_probs.
    """
    obs_list, action_list, logp_list, reward_list, done_list, value_list = [], [], [], [], [], []

    obs = torch.tensor(env.reset(), dtype=torch.float32).to(device)
    done = False
    steps = 0
    while not done and steps < max_steps:
        action, logp = policy.get_action(obs.unsqueeze(0), deterministic=False)
        value = policy.get_value(obs.unsqueeze(0))

        obs_next, reward, done, _ = env.step(action.item())

        obs_list.append(obs)
        action_list.append(action)
        logp_list.append(logp)
        reward_list.append(reward)
        done_list.append(done)
        value_list.append(value)

        obs = torch.tensor(obs_next, dtype=torch.float32).to(device)
        steps += 1

    # Convert to tensors
    obs_t = torch.stack(obs_list)
    actions_t = torch.stack(action_list).squeeze(-1)
    logp_t = torch.stack(logp_list).squeeze(-1)
    rewards_t = torch.tensor(reward_list, dtype=torch.float32).to(device)
    dones_t = torch.tensor(done_list, dtype=torch.float32).to(device)
    values_t = torch.stack(value_list).squeeze(-1)

    # Compute advantages and returns
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards_t))):
        delta = rewards_t[t] + gamma * (0 if dones_t[t] else values_t[t + 1] if t + 1 < len(values_t) else 0) - values_t[t]
        gae = delta + gamma * lam * (1 - dones_t[t]) * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = advantages + values_t

    return {
        "obs": obs_t,
        "actions": actions_t,
        "logp": logp_t,
        "advantages": advantages,
        "returns": returns,
    }


# ==============================
# Training functions
# ==============================
def train_pretrain(env, policy, optimizer, steps, device, buffer_path):
    """
    Pre‑train the policy on Phase 2 using PPO.
    """
    policy.train()
    epochs = 200
    batch_size = 32
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    for epoch in range(epochs):
        trajectories = []
        for _ in range(batch_size):
            traj = rollout(policy, env, device, env.max_steps, gamma, lam)
            trajectories.append(traj)

        # Flatten batches
        obs = torch.cat([t["obs"] for t in trajectories])
        actions = torch.cat([t["actions"] for t in trajectories])
        logp_old = torch.cat([t["logp"] for t in trajectories]).detach()
        advantages = torch.cat([t["advantages"] for t in trajectories]).detach()
        returns = torch.cat([t["returns"] for t in trajectories]).detach()

        # Policy loss
        logits, values = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp_new = dist.log_prob(actions)
        ratio = (logp_new - logp_old).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = dist.entropy().mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"[Pretrain] Epoch {epoch+1}/{epochs}  Loss {loss.item():.4f}")

    # Save pre‑trained weights
    torch.save(policy.state_dict(), "pretrained.pt")
    print("Pre‑training finished, weights saved to pretrained.pt")

    # Build buffer of states for BC/EM
    buffer = []
    preenv = AppleRetrievalEnv(phase="phase2")
    for _ in range(200):
        obs = preenv.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = policy.get_probs(obs_t.unsqueeze(0))
            action = torch.bernoulli(probs).long().item()
            obs, _, done, _ = preenv.step(action)
            buffer.append(obs)
    with open(buffer_path, "wb") as f:
        import pickle
        pickle.dump(buffer, f)
    print(f"Buffer of {len(buffer)} states stored to {buffer_path}")


def train_finetune(env, policy, optimizer, steps, device, method,
                   pretrained_policy, buffer, fisher, pre_params,
                   memory_buffer=None):
    """
    Fine‑tune on Phase 1 with a chosen knowledge‑retention method.
    """
    policy.train()
    pretrained_policy.eval()
    epochs = 200
    batch_size = 32
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    # Coefficients for auxiliary losses
    bc_coef = 1.0
    ks_coef = 1.0
    ewc_coef = 1e5
    em_coef = 5.0

    for epoch in range(epochs):
        trajectories = []
        bc_losses, ks_losses, ewc_losses, em_losses = [], [], [], []

        for _ in range(batch_size):
            traj = rollout(policy, env, device, env.max_steps, gamma, lam)
            trajectories.append(traj)

            # BC loss
            if method == "bc" and buffer:
                batch_states = random.sample(buffer, min(32, len(buffer)))
                batch_states = torch.tensor(batch_states, dtype=torch.float32).to(device)
                with torch.no_grad():
                    target_probs = pretrained_policy.get_probs(batch_states)
                current_probs = policy.get_probs(batch_states)
                bc_losses.append(kl_divergence(target_probs, current_probs).mean())

            # KS loss
            if method == "ks":
                online_states = traj["obs"]
                with torch.no_grad():
                    target_probs = pretrained_policy.get_probs(online_states)
                current_probs = policy.get_probs(online_states)
                ks_losses.append(kl_divergence(target_probs, current_probs).mean())

            # EWC loss
            if method == "ewc":
                for name, p in policy.named_parameters():
                    ewc_losses.append((fisher[name] * (p - pre_params[name]) ** 2).sum())

            # EM loss
            if method == "em" and memory_buffer:
                if len(memory_buffer) > 0:
                    sample = random.sample(memory_buffer, min(32, len(memory_buffer)))
                    sample_states = torch.tensor([s[0] for s in sample], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        target_probs = pretrained_policy.get_probs(sample_states)
                    current_probs = policy.get_probs(sample_states)
                    em_losses.append(kl_divergence(target_probs, current_probs).mean())

            # Store trajectories for EM
            if method == "em" and memory_buffer is not None:
                memory_buffer.extend([(s, a, r, d) for s, a, r, d in zip(
                    traj["obs"].cpu().numpy(),
                    traj["actions"].cpu().numpy(),
                    traj["advantages"].cpu().numpy(),  # not used but placeholder
                    traj["advantages"].cpu().numpy()   # placeholder
                )])

        # Flatten batches
        obs = torch.cat([t["obs"] for t in trajectories])
        actions = torch.cat([t["actions"] for t in trajectories])
        logp_old = torch.cat([t["logp"] for t in trajectories]).detach()
        advantages = torch.cat([t["advantages"] for t in trajectories]).detach()
        returns = torch.cat([t["returns"] for t in trajectories]).detach()

        # Policy loss
        logits, values = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp_new = dist.log_prob(actions)
        ratio = (logp_new - logp_old).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = dist.entropy().mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        # Add auxiliary losses
        if method == "bc" and bc_losses:
            loss += bc_coef * torch.stack(bc_losses).mean()
        if method == "ks" and ks_losses:
            loss += ks_coef * torch.stack(ks_losses).mean()
        if method == "ewc" and ewc_losses:
            loss += ewc_coef * torch.stack(ewc_losses).mean()
        if method == "em" and em_losses:
            loss += em_coef * torch.stack(em_losses).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"[Finetune] Epoch {epoch+1}/{epochs}  Loss {loss.item():.4f}")

    # Evaluate final success rate on Phase 1
    success = 0
    trials = 200
    policy.eval()
    for _ in range(trials):
        obs = torch.tensor(env.reset(), dtype=torch.float32).to(device)
        done = False
        while not done:
            action, _ = policy.get_action(obs.unsqueeze(0), deterministic=True)
            obs_next, _, done, _ = env.step(action.item())
            obs = torch.tensor(obs_next, dtype=torch.float32).to(device)
        if env.pos == env.apple_pos:
            success += 1
    success_rate = success / trials
    print(f"Final success rate on Phase 1: {success_rate*100:.1f}%")
    # Save results
    with open(f"results_{method}.json", "w") as f:
        json.dump({"success_rate": success_rate}, f)


# ==============================
# Main
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--method", type=str,
                        choices=["vanilla", "bc", "ewc", "ks", "em"],
                        default="vanilla")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper‑parameters
    steps = 1000  # not used directly

    if args.mode == "pretrain":
        env = AppleRetrievalEnv(phase="phase2")
        policy = ActorCritic().to(device)
        optimizer = optim.Adam(policy.parameters(), lr=3e-4)
        buffer_path = "pretrain_buffer.pkl"
        train_pretrain(env, policy, optimizer, steps, device, buffer_path)
    else:
        env = AppleRetrievalEnv(phase="phase1")
        policy = ActorCritic().to(device)
        optimizer = optim.Adam(policy.parameters(), lr=3e-4)

        # Load pre‑trained weights
        pretrained_policy = ActorCritic().to(device)
        pretrained_policy.load_state_dict(torch.load("pretrained.pt", map_location=device))

        # Build buffer for BC and EM
        buffer = None
        if args.method in ("bc", "em"):
            buffer_path = "pretrain_buffer.pkl"
            import pickle
            with open(buffer_path, "rb") as f:
                buffer = pickle.load(f)

        # Fisher for EWC
        fisher = None
        pre_params = None
        if args.method == "ewc":
            env_fisher = AppleRetrievalEnv(phase="phase2")
            fisher = compute_fisher(pretrained_policy, env_fisher, device)
            pre_params = {n: p.clone().detach() for n, p in pretrained_policy.named_parameters()}

        # Episodic memory buffer for EM
        memory_buffer = deque(maxlen=5000) if args.method == "em" else None

        train_finetune(env, policy, optimizer, steps, device,
                       args.method, pretrained_policy, buffer,
                       fisher, pre_params, memory_buffer)


if __name__ == "__main__":
    main()