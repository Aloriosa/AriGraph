import torch
import argparse
import numpy as np
from tqdm import tqdm
from fre.model import FREEncoder, FREDecoder
from fre.reward_prior import sample_reward_function
from fre.dataset import OfflineDataset

torch.manual_seed(0)
np.random.seed(0)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = OfflineDataset(args.dataset)
    state_dim = dataset.observations.shape[1]

    encoder = FREEncoder(state_dim).to(device)
    decoder = FREDecoder(state_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr
    )
    beta = args.beta
    K_enc = args.k_enc
    K_dec = args.k_dec

    for epoch in range(args.epochs):
        pbar = tqdm(range(args.steps_per_epoch), desc=f'Epoch {epoch+1}')
        for _ in pbar:
            # Sample a random reward function
            reward_fn = sample_reward_function(dataset.data, state_dim)

            # Encoder context
            idx_enc = torch.randint(0, dataset.length, (K_enc,))
            states_enc = dataset.observations[idx_enc].to(device)
            rewards_enc = reward_fn(states_enc).unsqueeze(-1).to(device)
            z, mean, logvar = encoder(states_enc.unsqueeze(0),
                                      rewards_enc.unsqueeze(0))  # batch=1

            # Decoder target
            idx_dec = torch.randint(0, dataset.length, (K_dec,))
            states_dec = dataset.observations[idx_dec].to(device)
            rewards_true = reward_fn(states_dec).unsqueeze(-1).to(device)

            preds = decoder(states_dec.unsqueeze(0), z)  # (1, K_dec)
            mse = ((preds - rewards_true.squeeze(-1)).pow(2)).mean()
            kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            loss = mse + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % args.checkpoint_every == 0:
            torch.save({'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict()},
                       f'{args.output_dir}/fre_epoch{epoch}.pt')
    torch.save({'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict()},
               f'{args.output_dir}/fre_final.pt')
    print(f'FRE training finished. Checkpoint in {args.output_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='antmaze-large-diverse-v2')
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--k_enc', type=int, default=32)
    parser.add_argument('--k_dec', type=int, default=8)
    parser.add_argument('--checkpoint_every', type=int, default=5)
    args = parser.parse_args()
    main(args)