import torch
import argparse
import numpy as np
from tqdm import tqdm
from fre.model import FREEncoder
from policy import PolicyNet, QNet
from fre.dataset import OfflineDataset
from fre.reward_prior import sample_reward_function

torch.manual_seed(0)
np.random.seed(0)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = OfflineDataset(args.dataset)
    state_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]

    # Load frozen FRE encoder
    encoder = FREEncoder(state_dim).to(device)
    encoder.load_state_dict(torch.load(args.fre_checkpoint)['encoder'])
    encoder.eval()

    policy = PolicyNet(state_dim, 32, action_dim).to(device)
    q1 = QNet(state_dim, action_dim, 32).to(device)
    q2 = QNet(state_dim, action_dim, 32).to(device)
    target_q1 = QNet(state_dim, action_dim, 32).to(device)
    target_q2 = QNet(state_dim, action_dim, 32).to(device)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    policy_opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)

    gamma = args.gamma
    for epoch in range(args.epochs):
        pbar = tqdm(range(args.steps_per_epoch), desc=f'Policy Epoch {epoch+1}')
        for _ in pbar:
            batch = dataset.sample_batch(args.batch_size)
            obs = batch['observations'].to(device)
            next_obs = batch['next_observations'].to(device)
            act = batch['actions'].to(device)

            # Random reward function for this batch
            reward_fn = sample_reward_function(dataset.data, state_dim)

            # Encode latent z from a small context set
            idx_ctx = torch.randint(0, dataset.length, (args.k_ctx,))
            states_ctx = dataset.observations[idx_ctx].to(device)
            rewards_ctx = reward_fn(states_ctx).unsqueeze(-1).to(device)
            with torch.no_grad():
                z, _, _ = encoder(states_ctx.unsqueeze(0), rewards_ctx.unsqueeze(0))
            z = z.squeeze(0)

            # Compute target Q
            with torch.no_grad():
                rewards = reward_fn(obs).to(device)
                next_z, _, _ = encoder(next_obs.unsqueeze(0),
                                       reward_fn(next_obs).unsqueeze(-1).unsqueeze(0))
                next_z = next_z.squeeze(0)

                mean_next, std_next = policy(next_obs, next_z)
                next_act = mean_next + std_next * torch.randn_like(mean_next)
                next_q1 = target_q1(next_obs, next_act, next_z)
                next_q2 = target_q2(next_obs, next_act, next_z)
                next_q = torch.min(next_q1, next_q2)
                target_q = rewards + gamma * next_q

            # Current Q estimates
            q1_pred = q1(obs, act, z)
            q2_pred = q2(obs, act, z)
            q_loss = ((q1_pred - target_q)**2 + (q2_pred - target_q)**2).mean()

            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            # Policy loss (simple advantage estimate)
            mean, std = policy(obs, z)
            logp = -0.5 * ((act - mean) / std).pow(2) - torch.log(std) - 0.5 * torch.log(2 * torch.pi)
            logp = logp.sum(-1)
            advantage = q1_pred - q1_pred.detach()
            policy_loss = -(logp * advantage).mean()

            policy_opt.zero_grad()
            policy_loss.backward()
            policy_opt.step()

            # Soft target update
            for target, online in zip(target_q1.parameters(), q1.parameters()):
                target.data.copy_(0.995 * target.data + 0.005 * online.data)
            for target, online in zip(target_q2.parameters(), q2.parameters()):
                target.data.copy_(0.995 * target.data + 0.005 * online.data)

        if epoch % args.checkpoint_every == 0:
            torch.save({
                'policy': policy.state_dict(),
                'q1': q1.state_dict(),
                'q2': q2.state_dict(),
                'target_q1': target_q1.state_dict(),
                'target_q2': target_q2.state_dict()
            }, f'{args.output_dir}/policy_epoch{epoch}.pt')
    torch.save({
        'policy': policy.state_dict(),
        'q1': q1.state_dict(),
        'q2': q2.state_dict(),
        'target_q1': target_q1.state_dict(),
        'target_q2': target_q2.state_dict()
    }, f'{args.output_dir}/policy_final.pt')
    print(f'Policy training finished. Checkpoint in {args.output_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='antmaze-large-diverse-v2')
    parser.add_argument('--fre_checkpoint', required=True)
    parser.add_argument('--output_dir', default='policy_checkpoints')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--steps_per_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--k_ctx', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--checkpoint_every', type=int, default=5)
    args = parser.parse_args()
    main(args)