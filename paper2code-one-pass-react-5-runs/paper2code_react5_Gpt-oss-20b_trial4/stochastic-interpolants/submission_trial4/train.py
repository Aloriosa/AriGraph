import argparse
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
import torchvision.transforms as T

from datasets.cifar10 import CIFAR10Inpaint, CIFAR10SuperRes
from models.unet import UNetVelocity
from utils import get_device, ensure_dir

def train_inpainting(device, epochs, batch_size, lr):
    dataset = CIFAR10Inpaint(root='data', train=True, transform=T.Compose([T.ToTensor()]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = UNetVelocity(in_channels=3, base_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        for x0, x1, mask in pbar:
            x0 = x0.to(device)
            x1 = x1.to(device)
            mask = mask.to(device)

            B, C, H, W = x0.shape
            t = torch.rand(B, device=device)  # uniform [0,1]

            alpha = 1 - t
            beta = t

            # Interpolant I_t = alpha * x0 + beta * x1
            I_t = alpha[:, None, None, None] * x0 + beta[:, None, None, None] * x1
            # derivative dot I_t = x1 - x0
            dotI = x1 - x0

            # Predict velocity
            b_pred = model(I_t, t, mask=mask)

            loss = criterion(b_pred, dotI)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

    ensure_dir('outputs/checkpoints')
    torch.save(model.state_dict(), 'outputs/checkpoints/velocity_inpaint.pth')
    print(f'Checkpoint saved to outputs/checkpoints/velocity_inpaint.pth')

def train_superres(device, epochs, batch_size, lr):
    dataset = CIFAR10SuperRes(root='data', train=True, transform=T.Compose([T.ToTensor()]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = UNetVelocity(in_channels=3, base_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        for low, high in pbar:
            low = low.to(device)
            high = high.to(device)

            B, C, H, W = low.shape
            t = torch.rand(B, device=device)

            alpha = 1 - t
            beta = t

            I_t = alpha[:, None, None, None] * low + beta[:, None, None, None] * high
            dotI = high - low

            b_pred = model(I_t, t, mask=None)

            loss = criterion(b_pred, dotI)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

    ensure_dir('outputs/checkpoints')
    torch.save(model.state_dict(), 'outputs/checkpoints/velocity_sr.pth')
    print(f'Checkpoint saved to outputs/checkpoints/velocity_sr.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--task', choices=['inpainting', 'superresolution'], required=True)
    args = parser.parse_args()

    device = get_device()
    print(f'Using device: {device}')

    if args.task == 'inpainting':
        train_inpainting(device, args.epochs, args.batch_size, args.lr)
    else:
        train_superres(device, args.epochs, args.batch_size, args.lr)

if __name__ == '__main__':
    main()