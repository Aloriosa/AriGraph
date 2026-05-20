import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import get_dataloader
from .model import create_velocity_model
from .utils import set_seed, get_time_embedding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'])
    set_seed(cfg['seed'])

    # Data
    train_loader = get_dataloader(
        cfg['dataset_path'], cfg['batch_size'], cfg['num_workers']
    )

    # Model
    model = create_velocity_model(cfg).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg['scheduler_gamma']
    )

    # Interpolant parameters
    alpha = lambda t: 1 - t
    beta = lambda t: t
    alpha_dot = lambda t: -1.0
    beta_dot = lambda t: 1.0
    sigma = cfg['sigma']

    # Training loop
    global_step = 0
    for epoch in range(cfg['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for batch_idx, (x1, _) in enumerate(pbar):
            x1 = x1.to(device)  # shape [B,3,H,W]
            B = x1.shape[0]
            t = torch.rand(B, device=device)  # uniform [0,1]
            # Base density: x0 = x1 + sigma * z
            z = torch.randn_like(x1)
            x0 = x1 + sigma * z

            # Interpolated sample
            I_t = alpha(t)[:, None, None, None] * x0 + \
                  beta(t)[:, None, None, None] * x1
            # True velocity
            vel_true = alpha_dot(t)[:, None, None, None] * x0 + \
                       beta_dot(t)[:, None, None, None] * x1

            # Predict velocity
            vel_pred = model(I_t, t)

            # Loss L_b
            loss = torch.mean(torch.sum(vel_pred**2, dim=[1,2,3])
                              - 2 * torch.sum(vel_true * vel_pred, dim=[1,2,3]))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
            optimizer.step()

            if global_step % cfg['scheduler_step'] == 0:
                scheduler.step()

            pbar.set_postfix(loss=loss.item())
            global_step += 1

        # Save checkpoint
        ckpt_path = f"checkpoints/ckpt_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()