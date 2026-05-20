import argparse
import csv
import torch
import torch.optim as optim
import numpy as np
from .policy import LinearPolicy
from .apple_env import AppleRetrieval
from .utils import kl_divergence

def evaluate(policy, env, phase=1, num_episodes=200):
    success = 0
    steps_list = []
    for _ in range(num_episodes):
        obs = env.reset(phase=phase)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        total_steps = 0
        done = False
        while not done and total_steps < 100:
            action = policy.get_action(obs_t)
            obs, reward, done, _ = env.step(action)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            total_steps += 1
        if reward > 0:
            success += 1
        steps_list.append(total_steps)
    return success / num_episodes, np.mean(steps_list)

def train_vanilla(policy, env, pretrain_phase, finetune_phase, epochs=200, lr=0.01):
    opt = optim.Adam(policy.parameters(), lr=lr)
    for epoch in range(epochs):
        # Pre‑train on phase 2 (M -> 0, move left)
        obs = env.reset(phase=2)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([[0.0]], dtype=torch.float32)  # action 0 (left)
        loss = F.binary_cross_entropy_with_logits(policy(obs_t), target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Fine‑tune on phase 1 (0 -> M, move right)
        obs = env.reset(phase=1)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([[1.0]], dtype=torch.float32)  # action 1 (right)
        loss = F.binary_cross_entropy_with_logits(policy(obs_t), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy

def train_bc(policy, env, pretrain_phase, finetune_phase, epochs=200, lr=0.01, bc_coef=1.0, buffer_size=100):
    opt = optim.Adam(policy.parameters(), lr=lr)
    # Build buffer from pre‑train phase
    buffer = []
    for _ in range(buffer_size):
        obs = env.reset(phase=2)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = policy.get_action(obs_t)
        buffer.append((obs_t, torch.tensor([[action]], dtype=torch.float32)))

    for epoch in range(epochs):
        # Pre‑train step
        obs = env.reset(phase=2)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([[0.0]], dtype=torch.float32)
        loss_pre = F.binary_cross_entropy_with_logits(policy(obs_t), target)

        # Fine‑tune step
        obs = env.reset(phase=1)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([[1.0]], dtype=torch.float32)
        loss_finetune = F.binary_cross_entropy_with_logits(policy(obs_t), target)

        # BC loss
        bc_loss = 0.0
        for s_t, a in buffer:
            logits = policy(s_t)
            loss = F.binary_cross_entropy_with_logits(logits, a)
            bc_loss += loss
        bc_loss /= len(buffer)

        loss = loss_pre + loss_finetune + bc_coef * bc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy

def main(args):
    env = AppleRetrieval(M=args.M)
    policy_vanilla = LinearPolicy()
    policy_bc = LinearPolicy()

    policy_vanilla = train_vanilla(policy_vanilla, env, pretrain_phase=2, finetune_phase=1, epochs=args.epochs, lr=args.lr)
    policy_bc = train_bc(policy_bc, env, pretrain_phase=2, finetune_phase=1, epochs=args.epochs, lr=args.lr, bc_coef=args.bc_coef)

    succ_vanilla, _ = evaluate(policy_vanilla, env, phase=1, num_episodes=args.eval_eps)
    succ_bc, _ = evaluate(policy_bc, env, phase=1, num_episodes=args.eval_eps)

    with open(args.save, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'success_rate'])
        writer.writerow(['vanilla', succ_vanilla])
        writer.writerow(['bc', succ_bc])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='apple_results.csv')
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bc_coef', type=float, default=1.0)
    parser.add_argument('--eval_eps', type=int, default=200)
    args = parser.parse_args()
    main(args)