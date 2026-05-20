import argparse
import torch
import torch.optim as optim
import numpy as np
from envs.two_state_mdp import TwoStateMDP
from models.policy import LinearPolicy

def policy_gradient(env, policy, optimizer, epochs=2000, gamma=0.99):
    """Standard REINFORCE with baseline 0."""
    for epoch in range(epochs):
        state = torch.tensor(env.reset(), dtype=torch.long)
        log_probs = []
        rewards = []
        done = False
        while not done:
            action, logp = policy.sample(state)
            next_state, reward, done, _ = env.step(action.item())
            log_probs.append(logp)
            rewards.append(reward)
            state = torch.tensor(next_state, dtype=torch.long)

        # compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for logp, G in zip(log_probs, returns):
            loss += -logp * G
        loss /= len(log_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, loss {loss.item():.3f}")

def evaluate(policy, env, n_episodes=100):
    total = 0
    for _ in range(n_episodes):
        state = torch.tensor(env.reset(), dtype=torch.long)
        done = False
        ep_ret = 0
        while not done:
            probs = policy.forward(state)
            action = (probs > 0.5).long()
            next_state, reward, done, _ = env.step(action.item())
            ep_ret += reward
            state = torch.tensor(next_state, dtype=torch.long)
        total += ep_ret
    return total / n_episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="pretrain.pt")
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()

    env = TwoStateMDP(start_state=1)  # pre‑train only on state 1
    policy = LinearPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    policy_gradient(env, policy, optimizer, epochs=args.epochs)

    torch.save(policy.state_dict(), args.output)
    print(f"Pre‑trained policy saved to {args.output}")

    # evaluate on full environment
    env_full = TwoStateMDP()
    mean_ret = evaluate(policy, env_full)
    print(f"Mean return on full env (after pre‑train): {mean_ret:.2f}")

if __name__ == "__main__":
    main()