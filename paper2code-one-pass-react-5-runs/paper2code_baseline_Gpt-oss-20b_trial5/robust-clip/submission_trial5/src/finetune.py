#!/usr/bin/env python3
"""
Fine‑tune CLIP vision encoder with FARE loss on CIFAR‑10.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from src.fare import fare_loss

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load pre‑trained CLIP
    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    clip.to(device)
    clip.eval()  # text encoder stays frozen

    # Only fine‑tune vision encoder
    encoder = clip.vision_model
    encoder.train()

    # Optimizer
    optimizer = optim.AdamW(encoder.parameters(), lr=1e-5, weight_decay=1e-4)

    # Data
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    train_set = torchvision.datasets.CIFAR10(root='data', train=True,
                                            download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch,
                              shuffle=True, num_workers=4, pin_memory=True)

    # Training loop
    encoder.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer.zero_grad()

            loss = fare_loss(encoder, imgs,
                             eps=args.eps,
                             pgd_steps=args.pgd_steps,
                             step_size=args.step_size)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}] '
                      f'Step [{i+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} finished, avg loss: {avg_loss:.4f}')

    # Save fine‑tuned encoder
    os.makedirs('models', exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join('models', 'fare_clip.pt'))
    print('Training finished. Fine‑tuned model saved to models/fare_clip.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of fine‑tuning epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--eps', type=float, default=4/255,
                        help='PGD epsilon')
    parser.add_argument('--pgd_steps', type=int, default=10,
                        help='Number of PGD steps')
    parser.add_argument('--step_size', type=float, default=1/255,
                        help='PGD step size')
    args = parser.parse_args()
    main(args)