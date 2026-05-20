import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import data
import reward_prior
import fre_encoder

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = data.SyntheticStateDataset(num_states=args.num_states, state_dim=args.state_dim)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = fre_encoder.FREModel(state_dim=args.state_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_mse = float('inf')
    rng = np.random.default_rng(12345)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_states in loader:
            batch_states = batch_states.to(device)  # (B, state_dim)
            B = batch_states.shape[0]

            # Sample K encoder states and K' decoder states
            idx_enc = rng.choice(len(dataset), size=(B, args.K), replace=True)
            idx_dec = rng.choice(len(dataset), size=(B, args.Kp), replace=True)

            enc_states = torch.stack([dataset[i] for i in idx_enc.flatten()]).view(B, args.K, -1).to(device)
            dec_states = torch.stack([dataset[i] for i in idx_dec.flatten()]).view(B, args.Kp, -1).to(device)

            # Sample random reward functions for each batch element
            enc_rewards = []
            dec_rewards = []
            for i in range(B):
                func, _ = reward_prior.sample_random_reward(enc_states[i].cpu().numpy(), rng)
                enc_rewards.append(func(enc_states[i].cpu().numpy()).reshape(-1,1))
                dec_rewards.append(func(dec_states[i].cpu().numpy()).reshape(-1,1))

            enc_rewards = torch.tensor(np.stack(enc_rewards), dtype=torch.float32, device=device)
            dec_rewards = torch.tensor(np.stack(dec_rewards), dtype=torch.float32, device=device)

            # Forward
            z, mu, logvar = model.encode(enc_states, enc_rewards)
            preds = model.decode(dec_states, z)

            # Losses
            recon_loss = F.mse_loss(preds, dec_rewards, reduction='mean')
            kl_loss = -(0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / (B * args.K)
            loss = recon_loss + args.beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs}  Loss: {avg_loss:.4f}")

    # Save model
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'fre_model.pt'))
    # Save config
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_states', type=int, default=2000,
                        help='Number of synthetic states to generate')
    parser.add_argument('--state_dim', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--K', type=int, default=32,
                        help='Number of encoder samples per batch element')
    parser.add_argument('--Kp', type=int, default=8,
                        help='Number of decoder samples per batch element')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.01,
                        help='KL weight in VIB objective')
    parser.add_argument('--out_dir', type=str, default='results',
                        help='Output directory')
    args = parser.parse_args()
    train(args)