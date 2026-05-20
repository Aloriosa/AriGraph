import argparse
import os
import random
import numpy as np
import gymnasium as gym
import torch

from envs.apple_retrieval import AppleRetrieval
from envs.two_state_mdp import TwoStateMDP
from algo.ppo import PPO
from algo.ewc import EWC
from algo.bc import BC
from algo.ks import KS
from algo.em import EM

def get_env(name):
    if name == "apple_retrieval":
        return AppleRetrieval(M=5, max_steps=100)
    elif name == "two_state_mdp":
        return TwoStateMDP()
    else:
        raise ValueError(f"Unknown env {name}")

def generate_rule_data(env, steps=20000):
    """
    Generate (obs, action) pairs using the rule:
    phase 0: right (action 1)
    phase 1: left (action 0)
    """
    obs, _ = env.reset()
    data = []
    for _ in range(steps):
        phase = int(obs[0] > 0.5)
        action = 1 if phase == 0 else 0
        next_obs, rew, done, _, _ = env.step(action)
        data.append((obs, action, rew, next_obs, done))
        obs = next_obs
        if done:
            obs, _ = env.reset()
    return data

def train_policy_on_rule(policy, data, lr=3e-4, epochs=5):
    """
    Train policy network to imitate the rule using cross‑entropy loss.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), 64):
            batch = data[i:i+64]
            obs_batch = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            act_batch = torch.tensor([b[1] for b in batch], dtype=torch.long)
            logits = policy(obs_batch)
            loss = torch.nn.functional.cross_entropy(logits, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def compute_fisher_from_data(policy, data_loader):
    ewc = EWC(policy)
    ewc.compute_fisher(data_loader)
    return ewc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="apple_retrieval")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--method", type=str, default="vanilla",
                        choices=["vanilla", "ewc", "bc", "ks", "em"])
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = get_env(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Policy network (same for pretrain and finetune)
    policy = PPO(obs_dim, act_dim).policy
    policy.to("cpu")

    if args.pretrain:
        print("=== Pre‑training ===")
        # Generate data from rule
        data = generate_rule_data(env, steps=args.max_steps)
        # Train policy to imitate rule
        train_policy_on_rule(policy, data, lr=3e-4, epochs=5)
        # Save teacher weights and buffer
        torch.save(policy.state_dict(), "teacher.pt")
        np.save("pretrain_buffer.npy", data, allow_pickle=True)
        print("Pre‑training completed. Teacher weights and buffer saved.")
        return

    # --- Fine‑tuning ---
    print("=== Fine‑tuning ===")
    # Load teacher weights
    if not os.path.exists("teacher.pt"):
        raise RuntimeError("Teacher weights not found. Run with --pretrain first.")
    policy.load_state_dict(torch.load("teacher.pt", map_location="cpu"))
    # Create a frozen copy of teacher for KL computations
    teacher_policy = PPO(obs_dim, act_dim).policy
    teacher_policy.load_state_dict(torch.load("teacher.pt", map_location="cpu"))
    teacher_policy.eval()
    for p in teacher_policy.parameters():
        p.requires_grad = False

    # Load pre‑train buffer
    if not os.path.exists("pretrain_buffer.npy"):
        raise RuntimeError("Pre‑training buffer not found. Run with --pretrain first.")
    pretrain_buffer = np.load("pretrain_buffer.npy", allow_pickle=True).tolist()

    # Wrap PPO with chosen method
    wrappers = []
    if args.method == "ewc":
        # Compute fisher on a DataLoader of (obs, action)
        data_loader = torch.utils.data.DataLoader(
            torch.tensor([[b[0] for b in pretrain_buffer], [b[1] for b in pretrain_buffer]],
                         dtype=torch.float32).t(),
            batch_size=64, shuffle=False)
        ewc = EWC(policy)
        ewc.compute_fisher(data_loader)
        wrappers.append(ewc)
    elif args.method == "bc":
        bc = BC(teacher_policy, pretrain_buffer, weight=2.0)
        wrappers.append(bc)
    elif args.method == "ks":
        ks = KS(teacher_policy, weight=0.5, decay=0.99998)
        wrappers.append(ks)
    elif args.method == "em":
        em = EM(teacher_policy, pretrain_buffer, batch_size=64, weight=1.0)
        wrappers.append(em)
    # else vanilla: no wrappers

    ppo = PPO(obs_dim, act_dim,
              lr=3e-4, gamma=0.99, clip=0.2,
              epochs=4, batch_size=64,
              vf_coef=0.5, ent_coef=0.01, device="cpu")

    # Use the pre‑trained policy as the initial policy for PPO
    ppo.policy = policy
    ppo.value = PPO(obs_dim, act_dim).value  # fresh value network

    total_steps = 0
    episode_rewards = []
    obs, _ = env.reset()
    ep_reward = 0
    while total_steps < args.max_steps:
        # Collect a minibatch of trajectories
        obs_batch, act_batch, rew_batch, logp_batch, val_batch, done_batch = ppo.collect_trajectories(
            env, steps=256)

        # Append dummy value for terminal state
        val_batch = np.append(val_batch, 0.0)
        adv, ret = ppo.compute_gae(rew_batch, val_batch, done_batch)
        # Convert to tensors
        wrapper_loss = 0.0
        if wrappers:
            # For BC and KS we need batch_obs
            batch_obs = torch.tensor(obs_batch, dtype=torch.float32)
            for w in wrappers:
                if isinstance(w, (BC, KS)):
                    wrapper_loss += w.loss(ppo.policy, teacher_policy, batch_obs)
                else:
                    wrapper_loss += w.loss(ppo.policy)

        # Update PPO with wrapper loss
        ppo.update(obs_batch, act_batch, logp_batch, ret, adv, wrapper_losses=wrapper_loss)

        # Logging
        total_steps += len(obs_batch)
        episode_rewards.append(np.sum(rew_batch))
        if total_steps % 1000 == 0:
            recent = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            avg_ret = np.mean(recent)
            print(f"Steps: {total_steps}  AvgReturn: {avg_ret:.2f}")

    # Final evaluation
    eval_eps = 10
    eval_rewards = []
    for _ in range(eval_eps):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            logits = ppo.policy(torch.tensor(obs, dtype=torch.float32))
            action = torch.argmax(logits).item()
            obs, rew, done, _, _ = env.step(action)
            ep_reward += rew
        eval_rewards.append(ep_reward)
    avg_ret = np.mean(eval_rewards)
    succ_rate = np.mean([1 if r > 0 else 0 for r in eval_rewards])
    print(f"Average return over {eval_eps} eval episodes: {avg_ret:.2f}")
    print(f"Success rate: {succ_rate:.2f}")

if __name__ == "__main__":
    main()