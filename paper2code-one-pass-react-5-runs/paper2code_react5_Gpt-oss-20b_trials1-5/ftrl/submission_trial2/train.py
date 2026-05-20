import argparse
import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import trange
from models import PolicyValueNet
from utils import (
    PenalisedCartPole,
    ReplayBuffer,
    kl_divergence,
    compute_fisher,
    AppleRetrieval,
)

# ---------- Hyper‑parameters ----------
DEFAULTS = {
    "env_name": "AppleRetrieval_finetune",
    "pretrain_steps": 2000,   # total timesteps for pre‑training
    "finetune_steps": 2000,   # total timesteps for fine‑tuning
    "batch_size": 64,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,          # GAE lambda
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "clip_eps": 0.2,
    "bc_weight": 1.0,
    "ewc_weight": 1e6,    # EWC penalty coefficient
    "ks_weight": 0.5,
    "buffer_capacity": 50000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "M": 10,
    "max_steps_per_episode": 100,
}

# ---------- Helpers ----------
def rollout(env, model, max_steps, device):
    """
    Run a single trajectory and return states, actions, rewards, dones, values.
    """
    states, actions, rewards, dones, values = [], [], [], [], []
    obs = env.reset()
    for _ in range(max_steps):
        state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(state_t)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        next_obs, reward, done, _ = env.step(action)

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())

        obs = next_obs
        if done:
            obs = env.reset()
    # Estimate final value for advantage calculation
    with torch.no_grad():
        final_state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, final_value = model(final_state_t)
    values.append(final_value.item())
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=np.bool_),
        np.array(values, dtype=np.float32),
    )

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """
    Compute GAE advantages and returns.
    """
    adv = np.zeros_like(rewards)
    ret = 0.0
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
        ret = rewards[t] + gamma * ret * (1.0 - dones[t])
    returns = adv + values[:-1]
    return adv, returns

def ppo_update(
    model,
    optimizer,
    states,
    actions,
    old_log_probs,
    advantages,
    returns,
    clip_eps,
    value_coef,
    entropy_coef,
    bc_loss=0.0,
    ks_loss=0.0,
    ewc_loss=0.0,
    em_loss=0.0,
    device="cpu",
):
    """
    Perform a single PPO update step (mini‑batch).
    """
    dataset_size = len(states)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    for start in range(0, dataset_size, DEFAULTS["batch_size"]):
        end = start + DEFAULTS["batch_size"]
        batch_idx = indices[start:end]

        batch_states = torch.tensor(states[batch_idx], dtype=torch.float32, device=device)
        batch_actions = torch.tensor(actions[batch_idx], dtype=torch.long, device=device)
        batch_old_logp = torch.tensor(old_log_probs[batch_idx], dtype=torch.float32, device=device)
        batch_adv = torch.tensor(advantages[batch_idx], dtype=torch.float32, device=device)
        batch_ret = torch.tensor(returns[batch_idx], dtype=torch.float32, device=device)

        logits, values = model(batch_states)
        log_probs = F.log_softmax(logits, dim=-1)
        batch_logp = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze()

        # PPO clipped surrogate loss
        ratio = torch.exp(batch_logp - batch_old_logp)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = -torch.min(ratio * batch_adv, clipped_ratio * batch_adv).mean()

        # Value loss
        value_loss = F.mse_loss(values, batch_ret)

        # Entropy bonus
        entropy = -(log_probs * torch.exp(log_probs)).sum(-1).mean()

        loss = (
            surrogate
            + value_coef * value_loss
            - entropy_coef * entropy
            + bc_loss
            + ks_loss
            + ewc_loss
            + em_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def collect_old_logp(
    model, states, actions, device
) -> np.ndarray:
    """
    Compute log‑probabilities of actions under the current policy.
    """
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    logits, _ = model(states_t)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(1, actions_t.unsqueeze(1)).squeeze().detach().cpu().numpy()

# ---------- Environment factory ----------
def make_env(name: str, args):
    """
    Create the appropriate gym environment based on the name.
    """
    if name == "AppleRetrieval_pretrain":
        return AppleRetrieval(start=args.M, M=args.M, max_steps=args.max_steps_per_episode)
    elif name == "AppleRetrieval_finetune":
        return AppleRetrieval(start=0, M=args.M, max_steps=args.max_steps_per_episode)
    else:
        # Default to CartPole if an unknown name is given
        return gym.make(name)

# ---------- Training / evaluation ----------
def train_pretrain(args):
    env = make_env("AppleRetrieval_pretrain", args)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device(args.device)
    model = PolicyValueNet(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    buffer = ReplayBuffer(capacity=args.buffer_capacity)

    t = 0
    while t < args.pretrain_steps:
        (
            states,
            actions,
            rewards,
            dones,
            values,
        ) = rollout(env, model, max_steps=args.max_steps_per_episode, device=device)

        # Store transitions for BC / EM
        for s, a in zip(states, actions):
            buffer.push(s, a)

        # Compute advantages & returns
        adv, returns = compute_gae(
            rewards, values, dones, args.gamma, args.lam
        )

        # Store old log‑probabilities for PPO
        old_logp = collect_old_logp(model, states, actions, device)

        # PPO update
        ppo_update(
            model,
            optimizer,
            states,
            actions,
            old_logp,
            adv,
            returns,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            device=device,
        )

        t += len(states)

    # Save artifacts
    torch.save(model.state_dict(), "policy_pretrain.pt")
    np.save("buffer_pretrain.npy", np.array(buffer.states))
    np.save("actions_pretrain.npy", np.array(buffer.actions))

    # Compute Fisher diagonal for EWC
    fisher = compute_fisher(model, env, num_samples=200, device=device, batch_size=args.batch_size)
    torch.save(fisher, "fisher_pretrain.pt")

    # Store baseline parameters for EWC
    baseline_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    torch.save(baseline_params, "baseline.pt")

    print("Pre‑training finished. Artifacts saved.")

def train_finetune(args):
    # Load pre‑trained policy
    device = torch.device(args.device)
    env = make_env(args.env_name, args)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = PolicyValueNet(obs_dim, action_dim).to(device)
    model.load_state_dict(torch.load("policy_pretrain.pt", map_location=device))

    # Load teacher (pre‑trained) policy for KL losses
    teacher = PolicyValueNet(obs_dim, action_dim).to(device)
    teacher.load_state_dict(torch.load("policy_pretrain.pt", map_location=device))
    teacher.eval()

    # Load Fisher for EWC
    if args.method == "ewc":
        fisher = torch.load("fisher_pretrain.pt", map_location=device)
    else:
        fisher = None

    # Load BC buffer
    bc_states = np.load("buffer_pretrain.npy")
    bc_actions = np.load("actions_pretrain.npy")
    bc_buffer = ReplayBuffer(capacity=len(bc_states))
    bc_buffer.states = bc_states.tolist()
    bc_buffer.actions = bc_actions.tolist()

    # Load baseline parameters for EWC
    if args.method == "ewc":
        baseline_params = torch.load("baseline.pt", map_location=device)
    else:
        baseline_params = None

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    t = 0
    while t < args.finetune_steps:
        (
            states,
            actions,
            rewards,
            dones,
            values,
        ) = rollout(env, model, max_steps=args.max_steps_per_episode, device=device)

        # Store online states for KS
        online_states = states
        online_actions = actions

        # Compute adv & returns
        adv, returns = compute_gae(
            rewards, values, dones, args.gamma, args.lam
        )

        # Store old log‑probabilities for PPO
        old_logp = collect_old_logp(model, states, actions, device)

        # Knowledge‑retention losses --------------------------------------------------
        bc_loss = torch.tensor(0.0, device=device)
        ks_loss = torch.tensor(0.0, device=device)
        ewc_loss = torch.tensor(0.0, device=device)
        em_loss = torch.tensor(0.0, device=device)

        # BC: KL over buffer states
        if args.method == "bc" and len(bc_buffer) > 0:
            s, a = bc_buffer.sample(args.batch_size)
            s_t = torch.tensor(s, dtype=torch.float32, device=device)
            logits, _ = model(s_t)
            with torch.no_grad():
                teacher_logits, _ = teacher(s_t)
            bc_loss = kl_divergence(teacher_logits, logits)

        # KS: KL over online states (sample a subset to keep memory usage low)
        if args.method == "ks" and len(online_states) > 0:
            idx = np.random.choice(len(online_states), args.batch_size, replace=False)
            s_subset = np.array([online_states[i] for i in idx], dtype=np.float32)
            s_t = torch.tensor(s_subset, dtype=torch.float32, device=device)
            logits, _ = model(s_t)
            with torch.no_grad():
                teacher_logits, _ = teacher(s_t)
            ks_loss = kl_divergence(teacher_logits, logits)

        # EWC: penalty on parameters
        if args.method == "ewc" and fisher is not None and baseline_params is not None:
            for name, p in model.named_parameters():
                if name in fisher:
                    ewc_loss += (fisher[name] * (p - baseline_params[name])**2).sum()

        # EM: mix buffer and online states in the loss
        if args.method == "em" and len(bc_buffer) > 0:
            # Sample half from buffer, half from online
            s_bc, a_bc = bc_buffer.sample(args.batch_size // 2)
            idx_online = np.random.choice(len(online_states), args.batch_size // 2, replace=False)
            s_online = np.array([online_states[i] for i in idx_online], dtype=np.float32)
            a_online = np.array([online_actions[i] for i in idx_online], dtype=np.int64)

            s_mix = np.concatenate([s_bc, s_online], axis=0)
            a_mix = np.concatenate([a_bc, a_online], axis=0)

            s_t = torch.tensor(s_mix, dtype=torch.float32, device=device)
            a_t = torch.tensor(a_mix, dtype=torch.long, device=device)
            logits, _ = model(s_t)
            em_loss = F.cross_entropy(logits, a_t)

        # Convert losses to weighted terms
        bc_loss = args.bc_weight * bc_loss
        ks_loss = args.ks_weight * ks_loss
        ewc_loss = args.ewc_weight * ewc_loss
        # em_loss already includes the correct weighting through batch size

        # PPO update with additional losses
        ppo_update(
            model,
            optimizer,
            states,
            actions,
            old_logp,
            adv,
            returns,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            bc_loss=bc_loss,
            ks_loss=ks_loss,
            ewc_loss=ewc_loss,
            em_loss=em_loss,
            device=device,
        )

        t += len(states)

    torch.save(model.state_dict(), f"policy_finetune_{args.method}.pt")
    print(f"Fine‑tuning ({args.method}) finished. Model saved.")

def eval_policy(args, start_state=None):
    env = make_env(args.env_name, args)
    if start_state is not None:
        # Reset to a specific start state
        env.reset()
        env.pos = start_state
        env.phase = 1 if start_state < args.M else 2
        env.steps = 0
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device(args.device)
    model = PolicyValueNet(obs_dim, action_dim).to(device)
    model.load_state_dict(torch.load(args.policy, map_location=device))
    model.eval()

    returns = []
    for _ in range(100):
        obs = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(state_t)
            action = torch.argmax(logits, dim=-1).item()
            obs, reward, done, _ = env.step(action)
            ep_ret += reward
        returns.append(ep_ret)

    mean_ret = np.mean(returns)
    print(f"Eval | policy: {os.path.basename(args.policy)} | mean reward: {mean_ret:.2f}")
    with open("results.txt", "w") as f:
        f.write(f"Mean reward: {mean_ret:.2f}\n")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["pretrain", "finetune", "eval", "eval_far"])
    parser.add_argument("--method", type=str, default="vanilla",
                        choices=["vanilla", "bc", "ewc", "ks", "em"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_name", type=str, default="AppleRetrieval_finetune")
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--max_steps_per_episode", type=int, default=100)
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Override defaults
    for k, v in DEFAULTS.items():
        setattr(args, k, v)

    if args.mode == "pretrain":
        train_pretrain(args)
    elif args.mode == "finetune":
        train_finetune(args)
    elif args.mode == "eval":
        eval_policy(args)
    elif args.mode == "eval_far":
        # Evaluate on FAR states by starting from position M
        eval_policy(args, start_state=args.M)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()