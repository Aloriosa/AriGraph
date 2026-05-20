import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import data
import reward_prior
import fre_encoder

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = data.SyntheticStateDataset(num_states=args.num_states, state_dim=args.state_dim)
    all_states = torch.tensor(dataset.states, dtype=torch.float32).to(device)

    model = fre_encoder.FREModel(state_dim=args.state_dim).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    rng = np.random.default_rng(42)

    # Sample a new reward function for evaluation
    func, _ = reward_prior.sample_random_reward(all_states.cpu().numpy(), rng)

    # Pick K encoder samples
    idx_enc = rng.choice(len(dataset), size=args.K, replace=True)
    enc_states = torch.tensor([dataset[i] for i in idx_enc], dtype=torch.float32).unsqueeze(0).to(device)
    enc_rewards = torch.tensor(func(enc_states.squeeze(0).cpu().numpy()).reshape(-1,1),
                               dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        z, _, _ = model.encode(enc_states, enc_rewards)

    # Predict rewards for all states
    preds = model.decode(all_states.unsqueeze(0), z).squeeze(0).cpu().numpy()

    true_rewards = func(all_states.cpu().numpy()).astype(np.float32)

    mse = np.mean((preds - true_rewards) ** 2)
    print(f"Evaluation MSE on held‑out states: {mse:.4f}")

    # Save report
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    report = {
        'mse': float(mse),
        'num_enc_samples': args.K,
        'reward_type': 'random',
    }
    with open(os.path.join(args.out_dir, 'eval_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained FRE model')
    parser.add_argument('--num_states', type=int, default=2000,
                        help='Number of synthetic states in dataset')
    parser.add_argument('--state_dim', type=int, default=2)
    parser.add_argument('--K', type=int, default=32,
                        help='Number of encoder samples for evaluation')
    parser.add_argument('--out_dir', type=str, default='results',
                        help='Output directory')
    args = parser.parse_args()
    evaluate(args)