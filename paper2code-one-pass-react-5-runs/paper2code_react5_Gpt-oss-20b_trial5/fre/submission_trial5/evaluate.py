import torch
import argparse
import json
import gymnasium
import numpy as np
from fre.model import FREEncoder
from policy import PolicyNet
from fre.dataset import OfflineDataset
from fre.reward_prior import sample_reward_function

torch.manual_seed(0)
np.random.seed(0)

def rollout(env, policy, z, max_steps=2000):
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        mean, std = policy(obs_t, z)
        action = (mean + std * torch.randn_like(mean)).cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = OfflineDataset(args.dataset)
    state_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]

    # Load FRE encoder
    encoder = FREEncoder(state_dim).to(device)
    encoder.load_state_dict(torch.load(args.fre_checkpoint)['encoder'])
    encoder.eval()

    # Load policy
    policy_state = torch.load(args.policy_checkpoint)['policy']
    policy = PolicyNet(state_dim, 32, action_dim).to(device)
    policy.load_state_dict(policy_state)
    policy.eval()

    # Create environment
    env = gymnasium.make(args.env_id, render_mode=None)

    # Sample a random reward function for evaluation
    reward_fn = sample_reward_function(dataset.data, state_dim)

    # Build latent from 32 context points
    idx_ctx = torch.randint(0, dataset.length, (args.k_ctx,))
    states_ctx = dataset.observations[idx_ctx].to(device)
    rewards_ctx = reward_fn(states_ctx).unsqueeze(-1).to(device)
    with torch.no_grad():
        z, _, _ = encoder(states_ctx.unsqueeze(0), rewards_ctx.unsqueeze(0))
    z = z.squeeze(0)

    # Run multiple episodes
    returns = []
    for _ in range(args.episodes):
        ret = rollout(env, policy, z, max_steps=args.max_steps)
        returns.append(ret)
    avg_ret = np.mean(returns)
    print(f'Average return over {args.episodes} episodes: {avg_ret:.2f}')

    # Save to JSON
    with open(args.output_file, 'w') as f:
        json.dump({'average_return': float(avg_ret)}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='antmaze-large-diverse-v2')
    parser.add_argument('--fre_checkpoint', required=True)
    parser.add_argument('--policy_checkpoint', required=True)
    parser.add_argument('--env_id', default='AntMaze-v2')
    parser.add_argument('--k_ctx', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--output_file', default='evaluation_result.json')
    args = parser.parse_args()
    main(args)