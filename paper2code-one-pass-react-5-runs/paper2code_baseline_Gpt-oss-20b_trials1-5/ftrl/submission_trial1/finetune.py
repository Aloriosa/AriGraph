import argparse
import torch
import torch.optim as optim
import numpy as np
from envs.two_state_mdp import TwoStateMDP
from models.policy import LinearPolicy

# Helper for computing Fisher (EWC)
def estimate_fisher(policy, env, samples=1000, device="cpu"):
    policy.eval()
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in policy.named_parameters()}

    for _ in range(samples):
        state = torch.tensor(env.reset(), dtype=torch.long, device=device)
        action, logp = policy.sample(state)
        grads = torch.autograd.grad(logp, policy.parameters(), retain_graph=True)
        for g, (n, p) in zip(grads, policy.named_parameters()):
            fisher[n] += g.detach() ** 2
    for n in fisher:
        fisher[n] /= samples
    return fisher

def policy_gradient(env, policy, optimizer, epochs=2000, gamma=0.99,
                    bc_buffer=None, bc_coef=0.0, ewc_fisher=None, ewc_coef=0.0):
    policy.train()
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

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for logp, G in zip(log_probs, returns):
            loss += -logp * G
        loss /= len(log_probs)

        # BC loss
        if bc_buffer is not None and bc_coef > 0:
            bc_states = torch.tensor([s for s, a in bc_buffer], dtype=torch.long)
            bc_actions = torch.tensor([a for s, a in bc_buffer], dtype=torch.long)
            probs = policy.forward(bc_states)
            dist = torch.distributions.Bernoulli(probs)
            bc_loss = -dist.log_prob(bc_actions).mean()
            loss += bc_coef * bc_loss

        # EWC loss
        if ewc_fisher is not None and ewc_coef > 0:
            ewc_loss = 0
            for n, p in policy.named_parameters():
                ewc_loss += (ewc_fisher[n] * (p - p.detach()) ** 2).sum()
            loss += ewc_coef * ewc_loss

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
    parser.add_argument("--pretrain", default="pretrain.pt")
    parser.add_argument("--output", default="finetuned.pt")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--bc", action="store_true", help="use behavioral cloning")
    parser.add_argument("--ewc", action="store_true", help="use EWC")
    parser.add_argument("--bc_coef", type=float, default=2.0)
    parser.add_argument("--ewc_coef", type=float, default=1e6)
    args = parser.parse_args()

    env = TwoStateMDP()
    policy = LinearPolicy()
    policy.load_state_dict(torch.load(args.pretrain))
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    # Build BC buffer from pre‑train data
    bc_buffer = None
    if args.bc:
        # Replay 2000 steps from pre‑train policy on full env
        env_full = TwoStateMDP()
        bc_buffer = []
        state = torch.tensor(env_full.reset(), dtype=torch.long)
        for _ in range(2000):
            action, _ = policy.sample(state)
            bc_buffer.append((state.item(), action.item()))
            next_state, _, done, _ = env_full.step(action.item())
            state = torch.tensor(next_state, dtype=torch.long)
            if done:
                state = torch.tensor(env_full.reset(), dtype=torch.long)

    # EWC fisher if needed
    ewc_fisher = None
    if args.ewc:
        ewc_fisher = estimate_fisher(policy, TwoStateMDP(start_state=1),
                                     samples=500)

    policy_gradient(env, policy, optimizer, epochs=args.epochs,
                    bc_buffer=bc_buffer,
                    bc_coef=args.bc_coef,
                    ewc_fisher=ewc_fisher,
                    ewc_coef=args.ewc_coef)

    torch.save(policy.state_dict(), args.output)
    print(f"Fine‑tuned policy saved to {args.output}")

    # evaluate before and after fine‑tuning
    mean_ret = evaluate(policy, env)
    print(f"Mean return on full env (after finetune): {mean_ret:.2f}")

if __name__ == "__main__":
    main()