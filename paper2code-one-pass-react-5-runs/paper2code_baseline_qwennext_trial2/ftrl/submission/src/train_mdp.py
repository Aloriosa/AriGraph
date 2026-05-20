import argparse
import csv
import torch
import torch.optim as optim
import numpy as np
from .policy import LinearPolicy
from .mdp_env import TwoStateMDP
from .utils import kl_divergence

def evaluate(policy, env, start_state=0, num_episodes=100):
    success = 0
    steps = []
    for _ in range(num_episodes):
        state = env.reset(start_state=start_state)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = policy.get_action(state_t)
        next_state, reward, done, _ = env.step(action)
        steps.append(1)  # one step per episode
        if reward > 0:
            success += 1
    return success / num_episodes, np.mean(steps)

def train_vanilla(policy, env, pretrain_phase, finetune_phase, epochs=200, lr=0.01):
    opt = optim.Adam(policy.parameters(), lr=lr)
    for epoch in range(epochs):
        # Pre‑train on pre‑train_phase (state 1)
        state = env.reset(start_state=1)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = policy.get_action(state_t)
        # We want action 0 (stay) on state 1
        target = torch.tensor([[0.0]], dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(policy(state_t), target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Fine‑tune on finetune_phase (state 0)
        state = env.reset(start_state=0)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = policy.get_action(state_t)
        # We want action 1 (move) on state 0
        target = torch.tensor([[1.0]], dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(policy(state_t), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy

def train_bc(policy, env, pretrain_phase, finetune_phase, epochs=200, lr=0.01, bc_coef=1.0, buffer_size=50):
    opt = optim.Adam(policy.parameters(), lr=lr)
    # Collect buffer from pre‑train phase
    buffer = []
    for _ in range(buffer_size):
        state = env.reset(start_state=1)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = policy.get_action(state_t)
        buffer.append((state_t, torch.tensor([[action]], dtype=torch.float32)))

    for epoch in range(epochs):
        # Pre‑train step (same as vanilla)
        state = env.reset(start_state=1)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([[0.0]], dtype=torch.float32)
        loss_pre = F.binary_cross_entropy_with_logits(policy(state_t), target)

        # Fine‑tune step
        state = env.reset(start_state=0)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([[1.0]], dtype=torch.float32)
        loss_finetune = F.binary_cross_entropy_with_logits(policy(state_t), target)

        # BC loss on buffer
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
    env = TwoStateMDP()
    policy_vanilla = LinearPolicy()
    policy_bc = LinearPolicy()

    # Train vanilla
    policy_vanilla = train_vanilla(policy_vanilla, env, pretrain_phase=1, finetune_phase=0, epochs=args.epochs, lr=args.lr)
    # Train with BC
    policy_bc = train_bc(policy_bc, env, pretrain_phase=1, finetune_phase=0, epochs=args.epochs, lr=args.lr, bc_coef=args.bc_coef)

    # Evaluate
    succ_vanilla, _ = evaluate(policy_vanilla, env, start_state=0, num_episodes=args.eval_eps)
    succ_bc, _ = evaluate(policy_bc, env, start_state=0, num_episodes=args.eval_eps)

    # Write CSV
    with open(args.save, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'success_rate'])
        writer.writerow(['vanilla', succ_vanilla])
        writer.writerow(['bc', succ_bc])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='mdp_results.csv')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bc_coef', type=float, default=1.0)
    parser.add_argument('--eval_eps', type=int, default=200)
    args = parser.parse_args()
    main(args)